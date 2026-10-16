import re

with open("models/attention_cores.py", "r", encoding="utf-8") as f:
    code = f.read()

# Replace the call method of LogLinearCore
start_idx = code.find("    def call(self, q_src, kv_src, training=None):")
end_idx = code.find("    def get_config(self):", start_idx)

new_call = """    def call(self, q_src, kv_src, training=None):
        orig_dtype = q_src.dtype
        B, H_q, W_q = tf.shape(q_src)[0], tf.shape(q_src)[1], tf.shape(q_src)[2]
        H_k, W_k = tf.shape(kv_src)[1], tf.shape(kv_src)[2]
        Q = tf.cast(self.W_q(q_src), tf.float32)
        K = tf.cast(self.W_k(kv_src), tf.float32)
        V = tf.cast(self.W_v(kv_src), tf.float32)
        Q = tf.reshape(Q, [B, H_q, W_q, self.num_heads, self.head_dim])
        K = tf.reshape(K, [B, H_k, W_k, self.num_heads, self.head_dim])
        V = tf.reshape(V, [B, H_k, W_k, self.num_heads, self.head_dim])
        Q = tf.math.l2_normalize(Q, axis=-1)
        K = tf.math.l2_normalize(K, axis=-1)
        KV = tf.expand_dims(K, -1) * tf.expand_dims(V, -2)
        KV = tf.reshape(KV, [B, H_k, W_k, self.num_heads * self.head_dim * self.head_dim])
        
        static_shape = kv_src.shape
        if static_shape[1] is not None and static_shape[2] is not None:
            max_dim = max(static_shape[1], static_shape[2])
            import math
            L = int(math.ceil(math.log2(max_dim))) if max_dim > 0 else self.max_L
        else:
            L = self.max_L
            
        pad_size = 2 ** L
        pad_h = pad_size - H_k
        pad_w = pad_size - W_k
        KV_padded = tf.pad(KV, [[0,0], [0, pad_h], [0, pad_w], [0,0]])
        mask = tf.ones([B, H_k, W_k, 1], dtype=tf.float32)
        mask_padded = tf.pad(mask, [[0,0], [0, pad_h], [0, pad_w], [0,0]])
        
        sum_KV_levels = []
        count_levels = []
        current_KV = KV_padded
        current_mask = mask_padded
        
        for l in range(L + 1):
            pool_factor = 2 ** l
            up_KV = tf.repeat(tf.repeat(current_KV, pool_factor, axis=1), pool_factor, axis=2)
            up_mask = tf.repeat(tf.repeat(current_mask, pool_factor, axis=1), pool_factor, axis=2)
            sum_KV_levels.append(up_KV)
            count_levels.append(up_mask)
            
            if l < L:
                current_KV = tf.nn.avg_pool2d(current_KV, 2, 2, 'VALID') * 4.0
                current_mask = tf.nn.avg_pool2d(current_mask, 2, 2, 'VALID') * 4.0

        lambdas = tf.nn.softplus(self.lambda_proj(q_src))
        lambdas = tf.reshape(lambdas, [B, H_q, W_q, self.num_heads, self.max_L + 1])
        lambdas = tf.cast(lambdas, tf.float32)

        out = tf.zeros([B, H_q, W_q, self.num_heads, self.head_dim], dtype=tf.float32)
        
        for l in range(L + 1):
            if l == 0:
                S_l = sum_KV_levels[0]
                count_S_l = count_levels[0]
            else:
                S_l = sum_KV_levels[l] - sum_KV_levels[l-1]
                count_S_l = count_levels[l] - count_levels[l-1]
                
            S_l = S_l[:, :H_q, :W_q, :]
            count_S_l = count_S_l[:, :H_q, :W_q, :]
            count_S_l = tf.maximum(count_S_l, 1.0)
            S_l = S_l / count_S_l
            
            S_l = tf.reshape(S_l, [B, H_q, W_q, self.num_heads, self.head_dim, self.head_dim])
            lam = tf.reshape(lambdas[:, :, :, :, l], [B, H_q, W_q, self.num_heads, 1])
            q_S_l = tf.einsum('bxyhd,bxyhde->bxyhe', Q, S_l)
            out = out + lam * q_S_l

        out = tf.reshape(out, [B, H_q, W_q, self.d_model])
        out = self.group_norm(out)
        out = tf.cast(out, orig_dtype)
        return self.W_o(out)

"""

new_code = code[:start_idx] + new_call + code[end_idx:]
with open("models/attention_cores.py", "w", encoding="utf-8") as f:
    f.write(new_code)
