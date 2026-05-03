import math
import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim=128, num_heads=8):
        super().__init__()
        assert embedding_dim % num_heads == 0

        self.mha = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim),
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, h):
        mha_out = self.mha(h, h, h, need_weights=False)[0]
        h = self.norm1(h + mha_out)

        ff_out = self.ff(h)
        h = self.norm2(h + ff_out)

        return h


class Decoder(nn.Module):
    def __init__(self, embedding_dim=128, num_heads=8, tanh_clipping=10.0):
        super().__init__()
        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.tanh_clipping = tanh_clipping

        # CVRP context = graph embedding + current node embedding + remaining capacity
        # => E + E + 1 = 2E + 1
        self.context_dim = embedding_dim * 2 + 1

        # 1) Multi-head glimpse layer
        self.project_query_glimpse = nn.Linear(self.context_dim, embedding_dim, bias=False)
        self.project_key_glimpse = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_val_glimpse = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_out_glimpse = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # 2) Final single-head output layer
        self.project_query_out = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_key_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def _get_context(self, embeddings, current_node, remaining_capacity):
        """
        embeddings: [B, N+1, E]
        current_node: [B]
        remaining_capacity: [B]
        return: [B, 2E+1]
        """
        batch_size = embeddings.size(0)

        graph_embedding = embeddings.mean(dim=1)  # [B, E]
        current_node_embedding = embeddings[
            torch.arange(batch_size, device=embeddings.device), current_node
        ]  # [B, E]
        capacity_feature = remaining_capacity.unsqueeze(-1)  # [B, 1]

        context = torch.cat(
            [graph_embedding, current_node_embedding, capacity_feature], dim=-1
        )
        return context

    def _split_heads(self, x):
        """
        x:
            [B, E]    -> query
            [B, N, E] -> keys / values
        return:
            if [B, E]    -> [B, H, 1, D]
            if [B, N, E] -> [B, H, N, D]
        """
        if x.dim() == 2:
            batch_size, embed_dim = x.shape
            x = x.view(batch_size, self.num_heads, self.head_dim)
            return x.unsqueeze(2)  # [B, H, 1, D]

        if x.dim() == 3:
            batch_size, num_nodes, embed_dim = x.shape
            x = x.view(batch_size, num_nodes, self.num_heads, self.head_dim)
            return x.permute(0, 2, 1, 3)  # [B, H, N, D]

        raise ValueError(f"Unsupported tensor shape: {tuple(x.shape)}")

    def _multi_head_glimpse(self, context, embeddings, mask):
        """
        context:    [B, 2E+1]
        embeddings: [B, N, E]
        mask:       [B, N]
        return:
            glimpse: [B, E]
        """
        batch_size, num_nodes, embed_dim = embeddings.shape

        # Eq. (5): q(c), k_i, v_i
        q = self.project_query_glimpse(context)    # [B, E]
        k = self.project_key_glimpse(embeddings)   # [B, N, E]
        v = self.project_val_glimpse(embeddings)   # [B, N, E]

        # Split multi-head
        q = self._split_heads(q)  # [B, H, 1, D]
        k = self._split_heads(k)  # [B, H, N, D]
        v = self._split_heads(v)  # [B, H, N, D]

        # Eq. (6): compatibility
        compatibility = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # [B, H, 1, N]

        if mask is not None:
            compatibility = compatibility.masked_fill(mask[:, None, None, :], float("-inf"))

        # Appendix A Eq. (12)
        attn = torch.softmax(compatibility, dim=-1)  # [B, H, 1, N]

        # Appendix A Eq. (13)
        heads = torch.matmul(attn, v)  # [B, H, 1, D]

        # Appendix A Eq. (14)
        heads = heads.squeeze(2).contiguous().view(batch_size, embed_dim)  # [B, E]
        glimpse = self.project_out_glimpse(heads)  # [B, E]
        return glimpse

    def _compute_output_logits(self, glimpse, embeddings, mask):
        """
        glimpse:    [B, E]
        embeddings: [B, N, E]
        mask:       [B, N]
        return:
            logits: [B, N]
        """
        q = self.project_query_out(glimpse)   # [B, E]
        k = self.project_key_out(embeddings)  # [B, N, E]

        # Final layer: single head
        compatibility = torch.matmul(q.unsqueeze(1), k.transpose(1, 2)).squeeze(1)
        compatibility = compatibility / math.sqrt(self.embedding_dim)  # [B, N]

        # Eq. (7): tanh clipping trước masking
        logits = self.tanh_clipping * torch.tanh(compatibility)

        if mask is not None:
            logits = logits.masked_fill(mask, float("-inf"))

        return logits

    def _compute_probs(self, glimpse, embeddings, mask):
        logits = self._compute_output_logits(glimpse, embeddings, mask)
        probs = torch.softmax(logits, dim=-1)
        return probs, logits

    def forward(self, embeddings, current_node, remaining_capacity, mask):
        """
        embeddings: [B, N, E]
        current_node: [B]
        remaining_capacity: [B]
        mask: [B, N]

        return:
            probs:  [B, N]
            logits: [B, N]
        """
        context = self._get_context(
            embeddings=embeddings,
            current_node=current_node,
            remaining_capacity=remaining_capacity,
        )

        glimpse = self._multi_head_glimpse(
            context=context,
            embeddings=embeddings,
            mask=mask,
        )

        probs, logits = self._compute_probs(
            glimpse=glimpse,
            embeddings=embeddings,
            mask=mask,
        )

        return probs, logits

    def select_node(self, probs, mask, decode_type="sampling"):
        """
        probs: [B, N]
        mask:  [B, N]
        return:
            selected: [B]
            log_prob: [B]
        """
        if decode_type == "greedy":
            selected = probs.argmax(dim=-1)
        elif decode_type == "sampling":
            selected = torch.multinomial(probs, num_samples=1).squeeze(1)

            # Phòng trường hợp hiếm do lỗi số học sample nhầm node bị mask
            invalid = mask.gather(1, selected.unsqueeze(1)).squeeze(1)
            while invalid.any():
                resampled = torch.multinomial(probs[invalid], num_samples=1).squeeze(1)
                selected[invalid] = resampled
                invalid = mask.gather(1, selected.unsqueeze(1)).squeeze(1)
        else:
            raise ValueError(f"Unknown decode_type: {decode_type}")

        selected_probs = probs.gather(1, selected.unsqueeze(1)).squeeze(1)
        log_prob = torch.log(selected_probs + 1e-12)
        return selected, log_prob


class CVRPModel(nn.Module):
    def __init__(self, embedding_dim=128, num_heads=8, num_layers=3):
        super().__init__()

        # Encoder
        self.init_embed_depot = nn.Linear(2, embedding_dim)
        self.init_embed_customers = nn.Linear(3, embedding_dim)

        self.encoder_layers = nn.ModuleList([
            AttentionLayer(embedding_dim, num_heads)
            for _ in range(num_layers)
        ])

        # Decoder
        self.decoder = Decoder(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
        )

        self._init_parameters()

    def _init_parameters(self):
        # Theo paper: uniform(-1/sqrt(d), 1/sqrt(d)) cho các weight matrix
        for param in self.parameters():
            if param.dim() > 1:
                stdv = 1.0 / math.sqrt(param.size(-1))
                param.data.uniform_(-stdv, stdv)

    def _get_embeddings(self, locs, demands):
        depot_loc = locs[:, 0:1, :]                # [B, 1, 2]
        customer_locs = locs[:, 1:, :]             # [B, n, 2]
        customer_demands = demands[:, 1:].unsqueeze(-1)  # [B, n, 1]

        customer_features = torch.cat([customer_locs, customer_demands], dim=-1)  # [B, n, 3]

        depot_embedding = self.init_embed_depot(depot_loc)              # [B, 1, E]
        customer_embeddings = self.init_embed_customers(customer_features)  # [B, n, E]

        h = torch.cat([depot_embedding, customer_embeddings], dim=1)    # [B, n+1, E]

        for layer in self.encoder_layers:
            h = layer(h)

        return h

    def forward(self, env, decode_type="sampling"):
        """
        env: CVRPenv đã reset(batch_size)
        decode_type: "sampling" hoặc "greedy"
        return:
            tour_rewards: [B]  (reward âm, = -tour length)
            total_log_probs: [B]
        """
        state = env.get_state()
        embeddings = self._get_embeddings(state["locs"], state["demands"])

        log_probs = []
        tour_rewards = torch.zeros(env.batch_size, device=env.device)
        done = torch.zeros(env.batch_size, dtype=torch.bool, device=env.device)

        while not done.all():
            mask = env.get_mask()

            probs, logits = self.decoder(
                embeddings=embeddings,
                current_node=state["current_node"],
                remaining_capacity=state["remaining_capacity"],
                mask=mask,
            )

            action, step_log_prob = self.decoder.select_node(
                probs=probs,
                mask=mask,
                decode_type=decode_type,
            )

            state, reward, done = env.step(action)

            log_probs.append(step_log_prob)
            tour_rewards += reward  # reward = -distance

        total_log_probs = torch.stack(log_probs, dim=0).sum(dim=0)
        return tour_rewards, total_log_probs
