import tensorflow as tf
import math

class NaLaCore(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, lambda_q=3.0, tau=0.5, lambda_k=3.0, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.lambda_q = lambda_q
        self.tau = tau
        self.lambda_k = lambda_k
        self.eps = eps

    def build(self, input_shape):
        self.cpe_q = tf.keras.layers.DepthwiseConv2D(3, padding='same', use_bias=False)
        self.cpe_kv = tf.keras.layers.DepthwiseConv2D(3, padding='same', use_bias=False)
        self.W_q = tf.keras.layers.Dense(self.d_model, use_bias=False)
        self.W_k = tf.keras.layers.Dense(self.d_model, use_bias=False)
        self.W_v = tf.keras.layers.Dense(self.d_model, use_bias=False)
        self.gate_proj = tf.keras.layers.Dense(self.d_model, use_bias=True)
        self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.W_o = tf.keras.layers.Dense(self.d_model, use_bias=False)
        super().build(input_shape)

    def _safe_pow(self, x, p):
        return tf.exp(p * tf.math.log(tf.abs(x) + 1e-6))

    def call(self, q_src, kv_src, training=None):
        orig_dtype = q_src.dtype
        q_src_cpe = q_src + self.cpe_q(q_src)
        kv_src_cpe = kv_src + self.cpe_kv(kv_src)
        Q = self.W_q(q_src_cpe)
        K = self.W_k(kv_src_cpe)
        V = self.W_v(kv_src_cpe)
        B, H_q, W_q = tf.shape(Q)[0], tf.shape(Q)[1], tf.shape(Q)[2]
        H_k, W_k = tf.shape(K)[1], tf.shape(K)[2]
        N_q, N_k = H_q * W_q, H_k * W_k
        Q = tf.reshape(Q, [B, N_q, self.num_heads, self.head_dim])
        K = tf.reshape(K, [B, N_k, self.num_heads, self.head_dim])
        V = tf.reshape(V, [B, N_k, self.num_heads, self.head_dim])
        Q_f32, K_f32, V_f32 = tf.cast(Q, tf.float32), tf.cast(K, tf.float32), tf.cast(V, tf.float32)
        norm_q = tf.norm(Q_f32, axis=-1, keepdims=True)
        dir_q = Q_f32 / (norm_q + self.eps)
        norm_k = tf.norm(K_f32, axis=-1, keepdims=True)
        dir_k = K_f32 / (norm_k + self.eps)
        pi_4 = tf.constant(math.pi / 4.0, dtype=tf.float32)
        theta_q = pi_4 * tf.math.tanh(dir_q)
        theta_k = pi_4 * tf.math.tanh(dir_k)
        f_q = self.lambda_q * (self.tau + tf.math.tanh(norm_q))
        pow_q = self._safe_pow(dir_q, f_q)
        pow_k = self._safe_pow(K_f32, self.lambda_k)
        phi_q = tf.concat([pow_q * tf.math.cos(theta_q), pow_q * tf.math.sin(theta_q)], axis=-1)
        phi_k = tf.concat([pow_k * tf.math.cos(theta_k), pow_k * tf.math.sin(theta_k)], axis=-1)
        S = tf.einsum('bnhi,bnhj->bhij', phi_k, V_f32)
        Z = tf.reduce_sum(phi_k, axis=1)
        out_num = tf.einsum('bnhi,bhij->bnhj', phi_q, S)
        out_den = tf.einsum('bnhi,bhi->bnh', phi_q, Z)
        out_den = tf.expand_dims(out_den, -1)
        out_num = tf.clip_by_value(out_num, -1e4, 1e4)
        out = out_num / (out_den + self.eps)
        out = tf.reshape(out, [B, H_q, W_q, self.d_model])
        out = tf.cast(out, orig_dtype)
        gate = tf.keras.activations.swish(self.gate_proj(q_src))
        out = self.ln(out) * gate
        return self.W_o(out)

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "num_heads": self.num_heads, "lambda_q": self.lambda_q, "tau": self.tau, "lambda_k": self.lambda_k, "eps": self.eps})
        return config

class LogLinearCore(tf.keras.layers.Layer):
    """
    chuyển thể 2D không nhân quả; quad-tree thay cho Fenwick tree 1D; không có gating/decay
    """
    def __init__(self, d_model, num_heads, max_L=10, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_L = max_L

    def build(self, input_shape):
        self.W_q = tf.keras.layers.Dense(self.d_model, use_bias=False)
        self.W_k = tf.keras.layers.Dense(self.d_model, use_bias=False)
        self.W_v = tf.keras.layers.Dense(self.d_model, use_bias=False)
        self.W_o = tf.keras.layers.Dense(self.d_model, use_bias=False)
        self.group_norm = tf.keras.layers.GroupNormalization(groups=self.num_heads)
        self.lambda_proj = tf.keras.layers.Dense(self.num_heads * (self.max_L + 1))
        super().build(input_shape)

    def call(self, q_src, kv_src, training=None):
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
        
        max_dim = tf.maximum(tf.cast(H_k, tf.float32), tf.cast(W_k, tf.float32))
        L_val = tf.cast(tf.math.ceil(tf.math.log(max_dim) / tf.math.log(2.0)), tf.int32)
        pad_size = tf.cast(tf.math.pow(2.0, tf.cast(L_val, tf.float32)), tf.int32)
        pad_h, pad_w = pad_size - H_k, pad_size - W_k
        KV_padded = tf.pad(KV, [[0,0], [0, pad_h], [0, pad_w], [0,0]])
        
        U_array = tf.TensorArray(tf.float32, size=self.max_L+1, dynamic_size=False, clear_after_read=False)
        
        def cond(l, current_KV, U_array): return l <= L_val
        def body(l, current_KV, U_array):
            pool_factor = tf.cast(tf.math.pow(2.0, tf.cast(l, tf.float32)), tf.int32)
            up = tf.repeat(tf.repeat(current_KV, pool_factor, axis=1), pool_factor, axis=2)
            U_array = U_array.write(l, up)
            next_KV = tf.cond(l < L_val, lambda: tf.nn.avg_pool2d(current_KV, 2, 2, 'VALID'), lambda: current_KV)
            return l + 1, next_KV, U_array

        _, _, U_array = tf.while_loop(cond, body, loop_vars=[0, KV_padded, U_array], maximum_iterations=self.max_L+1)
        
        lambdas = tf.nn.softplus(self.lambda_proj(q_src))
        lambdas = tf.reshape(lambdas, [B, H_q, W_q, self.num_heads, self.max_L + 1])
        lambdas = tf.cast(lambdas, tf.float32)
        
        def out_cond(l, out): return l <= L_val
        def out_body(l, out):
            U_l = U_array.read(l)
            def if_false():
                U_l_minus_1 = U_array.read(l-1)
                c_l = tf.cast(tf.math.pow(4.0, tf.cast(l, tf.float32)), tf.float32)
                c_l_minus_1 = tf.cast(tf.math.pow(4.0, tf.cast(l-1, tf.float32)), tf.float32)
                return (U_l * c_l - U_l_minus_1 * c_l_minus_1) / (c_l - c_l_minus_1)
            S_l = tf.cond(tf.equal(l, 0), lambda: U_l, if_false)
            S_l = S_l[:, :H_q, :W_q, :]
            S_l = tf.reshape(S_l, [B, H_q, W_q, self.num_heads, self.head_dim, self.head_dim])
            lam = tf.reshape(lambdas[:, :, :, :, l], [B, H_q, W_q, self.num_heads, 1])
            q_S_l = tf.einsum('bhwhd,bhwhde->bhwhe', Q, S_l)
            out = out + lam * q_S_l
            return l + 1, out

        out = tf.zeros([B, H_q, W_q, self.num_heads, self.head_dim], dtype=tf.float32)
        _, out = tf.while_loop(out_cond, out_body, loop_vars=[0, out], maximum_iterations=self.max_L+1)
        
        out = tf.reshape(out, [B, H_q, W_q, self.d_model])
        out = self.group_norm(out)
        out = tf.cast(out, orig_dtype)
        return self.W_o(out)

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "num_heads": self.num_heads, "max_L": self.max_L})
        return config

class MultipoleCore(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, w=8, L=3, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.w = w
        self.L = L

    def build(self, input_shape):
        self.D = tf.keras.layers.Conv2D(self.d_model, 2, strides=2, padding='valid')
        self.U = tf.keras.layers.Conv2DTranspose(self.d_model, 2, strides=2, padding='valid')
        self.W_q = tf.keras.layers.Dense(self.d_model, use_bias=False)
        self.W_k = tf.keras.layers.Dense(self.d_model, use_bias=False)
        self.W_v = tf.keras.layers.Dense(self.d_model, use_bias=False)
        self.W_o = tf.keras.layers.Dense(self.d_model, use_bias=False)
        super().build(input_shape)

    def window_attention(self, q, kv):
        B, H_q, W_q = tf.shape(q)[0], tf.shape(q)[1], tf.shape(q)[2]
        H_k, W_k = tf.shape(kv)[1], tf.shape(kv)[2]

        def global_attn():
            Q = tf.reshape(self.W_q(q), [B, H_q * W_q, self.num_heads, self.head_dim])
            K = tf.reshape(self.W_k(kv), [B, H_k * W_k, self.num_heads, self.head_dim])
            V = tf.reshape(self.W_v(kv), [B, H_k * W_k, self.num_heads, self.head_dim])
            Q, K, V = tf.transpose(Q, [0, 2, 1, 3]), tf.transpose(K, [0, 2, 1, 3]), tf.transpose(V, [0, 2, 1, 3])
            Q_f32, K_f32, V_f32 = tf.cast(Q, tf.float32), tf.cast(K, tf.float32), tf.cast(V, tf.float32)
            attn = tf.matmul(Q_f32, K_f32, transpose_b=True) / math.sqrt(self.head_dim)
            attn = tf.nn.softmax(attn, axis=-1)
            out = tf.matmul(attn, V_f32)
            out = tf.cast(tf.reshape(tf.transpose(out, [0, 2, 1, 3]), [B, H_q, W_q, self.d_model]), q.dtype)
            return out
            
        def local_attn():
            pad_h_q, pad_w_q = (self.w - H_q % self.w) % self.w, (self.w - W_q % self.w) % self.w
            q_pad = tf.pad(q, [[0,0], [0, pad_h_q], [0, pad_w_q], [0,0]])
            pad_h_k, pad_w_k = (self.w - H_k % self.w) % self.w, (self.w - W_k % self.w) % self.w
            kv_pad = tf.pad(kv, [[0,0], [0, pad_h_k], [0, pad_w_k], [0,0]])
            H_q_pad, W_q_pad = H_q + pad_h_q, W_q + pad_w_q
            H_k_pad, W_k_pad = H_k + pad_h_k, W_k + pad_w_k
            
            Q = tf.reshape(self.W_q(q_pad), [B, H_q_pad // self.w, self.w, W_q_pad // self.w, self.w, self.num_heads, self.head_dim])
            K = tf.reshape(self.W_k(kv_pad), [B, H_k_pad // self.w, self.w, W_k_pad // self.w, self.w, self.num_heads, self.head_dim])
            V = tf.reshape(self.W_v(kv_pad), [B, H_k_pad // self.w, self.w, W_k_pad // self.w, self.w, self.num_heads, self.head_dim])
            
            Q = tf.reshape(tf.transpose(Q, [0, 1, 3, 5, 2, 4, 6]), [-1, self.num_heads, self.w * self.w, self.head_dim])
            K = tf.reshape(tf.transpose(K, [0, 1, 3, 5, 2, 4, 6]), [-1, self.num_heads, self.w * self.w, self.head_dim])
            V = tf.reshape(tf.transpose(V, [0, 1, 3, 5, 2, 4, 6]), [-1, self.num_heads, self.w * self.w, self.head_dim])
            
            Q_f32, K_f32, V_f32 = tf.cast(Q, tf.float32), tf.cast(K, tf.float32), tf.cast(V, tf.float32)
            attn = tf.matmul(Q_f32, K_f32, transpose_b=True) / math.sqrt(self.head_dim)
            attn = tf.nn.softmax(attn, axis=-1)
            out = tf.matmul(attn, V_f32)
            out = tf.cast(out, q.dtype)
            out = tf.reshape(out, [B, H_q_pad // self.w, W_q_pad // self.w, self.num_heads, self.w, self.w, self.head_dim])
            out = tf.reshape(tf.transpose(out, [0, 1, 4, 2, 5, 3, 6]), [B, H_q_pad, W_q_pad, self.d_model])
            return out[:, :H_q, :W_q, :]

        return tf.cond(tf.logical_or(H_q <= self.w, W_q <= self.w), global_attn, local_attn)

    def call(self, q_src, kv_src, training=None):
        q_list, kv_list = [q_src], [kv_src]
        for l in range(self.L):
            q_list.append(self.D(q_list[-1]))
            kv_list.append(self.D(kv_list[-1]))
        attn_outputs = [self.window_attention(q_list[l], kv_list[l]) for l in range(self.L + 1)]
        out = attn_outputs[-1]
        for l in range(self.L - 1, -1, -1):
            out = self.U(out)
            tgt_shape = tf.shape(attn_outputs[l])
            out = out[:, :tgt_shape[1], :tgt_shape[2], :]
            out = out + attn_outputs[l]
        return self.W_o(out)

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "num_heads": self.num_heads, "w": self.w, "L": self.L})
        return config
