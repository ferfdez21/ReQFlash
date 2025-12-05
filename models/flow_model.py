
import torch
from torch import nn

from models.node_feature_net import NodeFeatureNet
from models.flash_ipa.edge_embedder import EdgeEmbedder

from models.flash_ipa.ipa import InvariantPointAttention as FlashInvariantPointAttention, EdgeTransition as FlashEdgeTransition, StructureModuleTransition as FlashStructureModuleTransition, BackboneUpdate as FlashBackboneUpdate
from models.flash_ipa.linear import Linear as FlashLinear
from models.flash_ipa.utils import check_config_ipa
from data import utils as du


class FlowModel(nn.Module):

    def __init__(self, model_conf):
        super(FlowModel, self).__init__()
        self._model_conf = model_conf
        self.mode = model_conf.mode
        self._ipa_conf = model_conf.ipa
        check_config_ipa(model_conf.ipa, model_conf.mode)
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE) 
        self.node_feature_net = NodeFeatureNet(model_conf.node_features)
        self.edge_embedder = EdgeEmbedder(model_conf.edge_features)

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = FlashInvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = FlashLinear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = FlashStructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = FlashBackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks-1:
                # No edge update on the last block.
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f'edge_transition_{b}'] = FlashEdgeTransition(
                    mode="2d" if self._model_conf.edge_features.mode == "orig_2d_bias" else "1d",
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._model_conf.edge_embed_size,
                    z_factor_rank=self._model_conf.ipa.z_factor_rank,
                )

    def forward(self, input_feats):
        node_mask = input_feats['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        diffuse_mask = input_feats['diffuse_mask']
        res_index = input_feats['res_idx']
        so3_t = input_feats['so3_t']
        r3_t = input_feats['r3_t']
        trans_t = input_feats['trans_t']
        rotquats_t = input_feats['rotquats_t']
        if 'trans_sc' not in input_feats:
            trans_sc = torch.zeros_like(trans_t)
        else:
            trans_sc = input_feats['trans_sc']

        # Initialize node and edge embeddings
        init_node_embed = self.node_feature_net(
            so3_t,
            r3_t,
            node_mask,
            diffuse_mask,
            res_index
        )
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        
        edge_embed, z_factor_1, z_factor_2, edge_mask = self.edge_embedder(
            init_node_embed,
            trans_t,
            trans_sc,
            node_mask,
            edge_embed=None,
            edge_mask=None,
        )
        
        # Initial rigids
        curr_rigids = du.create_rigid_quats(rotquats_t, trans_t)

        # Main trunk
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        edge_embed = edge_embed * edge_mask[..., None]
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,  # Should be None for flash attn mode
                z_factor_1,  # Should be None for not flash attn mode
                z_factor_2,  # Should be None for not flash attn mode
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]

            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).to(torch.bool))
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]

            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, (node_mask * diffuse_mask)[..., None])
            
            # Update edge embeddings
            if b < self._ipa_conf.num_blocks - 1:
                if self.mode == "orig_2d_bias":
                    edge_embed = self.trunk[f"edge_transition_{b}"](node_embed, edge_embed)  # edge_embed is B,L,L,D
                    edge_embed *= edge_mask[..., None]
                elif self.mode == "flash_1d_bias" or self.mode == "flash_2d_factorize_bias":
                    z_factor_1, z_factor_2 = self.trunk[f"edge_transition_{b}"](node_embed, None, z_factor_1, z_factor_2)
                    z_factor_1 *= node_mask[:, :, None, None]
                    z_factor_2 *= node_mask[:, :, None, None]
                else:
                    # no 2D bias
                    continue

        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans = curr_rigids.get_trans()
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()
        pred_rotquats = curr_rigids.get_rots().get_quats()
        return {
            'pred_trans': pred_trans,
            'pred_rotmats': pred_rotmats,
            'pred_rotquats': pred_rotquats,
        }
