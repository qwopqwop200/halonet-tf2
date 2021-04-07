import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from einops import rearrange, repeat

def expand_dim(t, dim, k):
    t = tf.expand_dims(t,axis = dim)
    expand_shape = [1] * len(t.shape)
    expand_shape[dim] = k
    return tf.tile(t,expand_shape)

def rel_to_abs(x):
    b, l, m = x.shape
    r = (m + 1) // 2

    col_pad = tf.zeros((b, l, 1))
    x = tf.concat((x, col_pad), axis = 2)
    flat_x = rearrange(x, 'b l c -> b (l c)')
    flat_pad = tf.zeros((b, m - l))
    flat_x_padded = tf.concat((flat_x, flat_pad), axis = 1)
    final_x = tf.reshape(flat_x_padded,(b, l + 1, m))
    final_x = final_x[:, :l, -r:]
    return final_x

def relative_logits_1d(q, rel_k):
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2

    logits = tf.einsum('b x y d, r d -> b x y r', q, rel_k)
    logits = rearrange(logits, 'b x y r -> (b x) y r')
    logits = rel_to_abs(logits)

    logits = tf.reshape(logits,(b, h, w, r))
    logits = expand_dim(logits, dim = 2, k = r)
    return logits

class RelPosEmb(layers.Layer):
    def __init__(self,block_size,rel_size,dim_head):
        super().__init__(trainable=True)
        height = width = rel_size
        scale = dim_head ** -0.5

        self.block_size = block_size
        inits = tf.random_normal_initializer()
        self.rel_height = tf.Variable(inits((height * 2 - 1, dim_head)) * scale,trainable=True)
        self.rel_width = tf.Variable(inits((width * 2 - 1, dim_head)) * scale, trainable=True)

    def call(self, q):
        block = self.block_size

        q = rearrange(q, 'b (x y) c -> b x y c', x = block)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b x i y j-> b (x y) (i j)')

        q = rearrange(q, 'b x y d -> b y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b x i y j -> b (y x) (j i)')
        return rel_logits_w + rel_logits_h

class HaloAttention(layers.Layer):
    def __init__(self,block_size,halo_size,dim_head = 64,heads = 8):
        super().__init__()
        assert halo_size > 0, 'halo size must be greater than 0'
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.block_size = block_size
        self.halo_size = halo_size

        self.inner_dim = dim_head * heads
        self.rel_pos_emb = RelPosEmb(block_size,block_size + (halo_size * 2),dim_head = dim_head)
        
    def build(self,shape):
        self.dim = shape[-1]
        self.to_q  = layers.Dense(self.inner_dim, use_bias = False)
        self.to_kv = layers.Dense(self.inner_dim * 2, use_bias = False)
        self.to_out = layers.Dense(self.dim)

    def call(self, x):
        x = tf.transpose(x,[0,3,1,2])
        b, c, h, w, block, halo, heads = *x.shape, self.block_size, self.halo_size, self.heads
        assert h % block == 0 and w % block == 0, 'fmap dimensions must be divisible by the block size'
        assert c == self.dim, f'channels for input ({c}) does not equal to the correct dimension ({self.dim})'

        q_inp = rearrange(x, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1 = block, p2 = block)
        
        kv_inp = tf.image.extract_patches(tf.transpose(x,[0,2,3,1]), 
                                          sizes = [1,block + halo * 2,block + halo * 2,1], 
                                          strides = [1,block,block,1], rates = [1,1,1,1],
                                          padding = 'SAME')
        kv_inp = tf.reshape(kv_inp,(kv_inp.shape[0],kv_inp.shape[3],-1))
        
        kv_inp = rearrange(kv_inp, 'b (c j) i -> (b i) j c', c = c)

        q = self.to_q(q_inp)
        k, v = tf.split(self.to_kv(kv_inp), 2, axis= -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = heads), (q, k, v))

        q *= self.scale

        sim = tf.einsum('b i d, b j d -> b i j', q, k)

        sim += self.rel_pos_emb(q)
        
        mask = tf.ones((1, h, w,1))
        mask = tf.image.extract_patches(mask, 
                                        sizes = [1,block + halo * 2,block + halo * 2,1], 
                                        strides = [1,block,block,1], rates = [1,1,1,1],
                                        padding = 'SAME')
        mask = tf.reshape(mask,(mask.shape[0],mask.shape[3],-1))
        
        mask = repeat(mask, '() j i -> (b i h) () j', b = b, h = heads)

        max_neg_value = -1e9
        sim = self.masked_fill(sim,mask, max_neg_value)
                                                                                                                 
        attn = keras.activations.softmax(sim,axis=-1)

        out = tf.einsum('b i j, b j d -> b i d', attn, v)

        out = rearrange(out, '(b h) n d -> b n (h d)', h = heads)
        out = self.to_out(out)
    
        out = rearrange(out, '(b h w) (p1 p2) c -> b c (h p1) (w p2)', b = b, h = (h // block), w = (w // block), p1 = block, p2 = block)
        out = tf.transpose(out,[0,2,3,1])
        return out
    
    def masked_fill(self,t, mask, value):
        return t * (1 - tf.cast(mask, tf.float32)) + value * tf.cast(mask, tf.float32)
      
def test():
    attn = HaloAttention(block_size = 8,halo_size = 4,dim_head = 64,heads = 4)
    f_map  = attn(tf.ones((1,32,32,512)))
