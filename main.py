import os
import pandas as pd
import numpy as np
import re
import pickle
import shutil
import zipfile

class Adam:

    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]

    def step(self):

        self.t += 1
        for i, param in enumerate(self.params):

            grad = param.grad
            grad = np.clip(grad, -2.5, 2.5)


            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad


            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)


            m_hat = self.m[i] / (1 - self.beta1 ** self.t)


            v_hat = self.v[i] / (1 - self.beta2 ** self.t)


            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):

        for param in self.params:
            param.grad = np.zeros_like(param.data)

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data,dtype=np.float64) if not isinstance(data, np.ndarray) else data
        self._children = _children
        self._op = _op
        self._prev = set(_children)
        self.grad = np.zeros_like(self.data,dtype=np.float64)
        self._backward = lambda: None
    def reduce_grad(grad, shape):
        while len(grad.shape) > len(shape):
           grad = grad.sum(axis=0)
        for i, dim in enumerate(shape):
           if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
        return grad
    @staticmethod
    def _match_shape(grad, shape):
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(shape):
            if dim == 1:
             grad = grad.sum(axis=i, keepdims=True)
        return grad
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += self._match_shape(out.grad, self.data.shape)
            other.grad += self._match_shape(out.grad, other.data.shape)

        out._backward = _backward
        return out
    def __mul__(self,other):
        other=other if isinstance(other,Value) else Value(other)
        out = Value(self.data*other.data,(self,other),'*')

        def _backward():
            self.grad += self._match_shape(out.grad * other.data, self.data.shape)
            other.grad += self._match_shape(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out
    def __matmul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data @ other.data, (self, other), '@')

        def _backward():
            grad = out.grad
            self.grad += self._match_shape(grad @ np.swapaxes(other.data, -1, -2), self.data.shape)
            other.grad += self._match_shape(np.swapaxes(self.data, -1, -2) @ grad, other.data.shape)

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), '-')

        def _backward():
            self.grad += self._match_shape(out.grad, self.data.shape)
            other.grad -= self._match_shape(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __pow__(self,other):
        assert isinstance(other,(int,float))
        out = Value(self.data**other,(self,),'**')

        def _backward():
            self.grad += other * self.data**(other-1) * out.grad

        out._backward = _backward
        return out
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, (self, other), '/')

        def _backward():
            self.grad += out.grad / other.data
            other.grad -= Value.reduce_grad(
            (self.data / (other.data ** 2)) * out.grad,
            other.grad.shape
             )

        out._backward = _backward
        return out
    def relu(self):
        out = Value(np.maximum(0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out
    def softmax(self):
        exps = np.exp(self.data - np.max(self.data, axis=-1, keepdims=True))
        out_data = exps / np.sum(exps, axis=-1, keepdims=True)
        out = Value(out_data, (self,), 'Softmax')

        def _backward():
            grad_out = out.grad
            self.grad += (grad_out - np.sum(grad_out * out.data, axis=-1, keepdims=True)) * out.data

        out._backward = _backward
        return out
    def exp(self):
        out_data = np.exp(self.data)
        out = Value(out_data, (self,), 'Exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out
    def log(self):
        out_data = np.log(self.data)
        out = Value(out_data, (self,), 'Log')

        def _backward():
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward
        return out
    def get_embedding(self, indices):

        emb = self.data[indices]


        out = Value(emb, (self,), 'Embedding')

        def _backward():

            np.add.at(self.grad, indices, out.grad)

        out._backward = _backward
        return out
    def layernorm(self, x, gamma, beta, eps=1e-5):

        mean = np.mean(x.data, axis=-1, keepdims=True)
        var = np.var(x.data, axis=-1, keepdims=True)

        normalized = (x.data - mean) / np.sqrt(var + eps)

        out_data = gamma.data * normalized + beta.data
        out = Value(out_data, (x, gamma, beta), 'LayerNorm')

        def _backward():
          N = x.data.shape[-1]
          grad_out = out.grad


          gamma.grad += np.sum(grad_out * normalized, axis=(0,1))
          beta.grad += np.sum(grad_out, axis=(0,1))

          grad_norm = grad_out * gamma.data

          grad_var = np.sum(
            grad_norm * (x.data - mean) * -0.5 * (var + eps) ** (-1.5),
            axis=-1, keepdims=True
         )

          grad_mean = np.sum(
            grad_norm * -1 / np.sqrt(var + eps),
            axis=-1, keepdims=True
          ) + grad_var * np.mean(-2 * (x.data - mean), axis=-1, keepdims=True)

          x.grad += (
            grad_norm / np.sqrt(var + eps)
            + grad_var * 2 * (x.data - mean) / N
            + grad_mean / N
         )

        out._backward = _backward
        return out
    # def crossentropy(self, target):
    #     assert self.data.shape == target.data.shape
    #     out_data = -np.sum(target.data * np.log(self.data + 1e-15), axis=-1)
    #     out = Value(out_data, (self, target), 'CrossEntropy')

    #     def _backward():
    #         grad = out.grad


    #         grad = np.expand_dims(grad, axis=-1)
    #         print(grad.shape)
    #         self.grad += -target.data / (self.data + 1e-15) * grad
    #         target.grad += -np.log(self.data + 1e-15) * grad

    #     out._backward = _backward
    #     return out

    def crossentropy_with_logits(self, target_value, label_smoothing=0.0):

        vocab_size = target_value.data.shape[-1]


        if label_smoothing > 0:
            smooth_target = (1 - label_smoothing) * target_value.data + (label_smoothing / vocab_size)
        else:
            smooth_target = target_value.data


        max_logits = np.max(self.data, axis=-1, keepdims=True)
        exps = np.exp(self.data - max_logits)
        probs = exps / np.sum(exps, axis=-1, keepdims=True)


        out_data = -np.sum(smooth_target * np.log(probs + 1e-15), axis=-1)


        out = Value(out_data, (self, target_value), 'CrossEntropyLogits')

        def _backward():

            grad_expanded = np.expand_dims(out.grad, axis=-1)
            self.grad += (probs - smooth_target) * grad_expanded

        out._backward = _backward
        return out
    def dropout(self, p=0.5):
        mask = (np.random.rand(*self.data.shape) > p).astype(float)
        out_data = self.data * mask / (1 - p)
        out = Value(out_data, (self,), 'Dropout')

        def _backward():
            self.grad += (mask / (1 - p)) * out.grad

        out._backward = _backward
        return out
    def sum(self, axis=None, keepdims=False):
        out=Value(np.sum(self.data,axis=axis,keepdims=keepdims),(self,),'Sum')
        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out
    def mean(self, axis=None, keepdims=False):
        out=Value(np.mean(self.data,axis=axis,keepdims=keepdims),(self,),'Mean')
        def _backward():
            self.grad += np.ones_like(self.data) * out.grad / self.data.size
        out._backward = _backward
        return out
    def T(self):
        out = Value(np.swapaxes(self.data, -1, -2), (self,), 'Transpose')

        def _backward():
          self.grad += np.swapaxes(out.grad, -1, -2)

        out._backward = _backward
        return out
    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (list, tuple)):
            axes = axes[0]
        axes = tuple(a % self.data.ndim if a < 0 else a for a in axes)
        out = Value(np.transpose(self.data, axes), (self,), 'Transpose')

        def _backward():
            # To invert the transpose, need the inverse permutation
            inv_axes = np.empty_like(axes)
            inv_axes[axes] = np.arange(len(axes))
            self.grad += np.transpose(out.grad, inv_axes)

        out._backward = _backward
        return out
    def reshape(self, *shape):
        out=Value(self.data.reshape(*shape),(self,),'Reshape')
        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out
    def sqrt(self):
        out=Value(np.sqrt(self.data),(self,),'Sqrt')
        def _backward():
            self.grad += 0.5 / np.sqrt(self.data) * out.grad
        out._backward = _backward
        return out
    def masked_fill(self, mask, value):
        out_data = np.where(mask.data, value, self.data)
        out = Value(out_data, (self, mask), 'MaskedFill')

        def _backward():
            self.grad += np.where(mask.data, 0, out.grad)
            mask.grad += np.sum(np.where(mask.data, 0, out.grad), axis=0)

        out._backward = _backward
        return out
    def split(self, num_splits, axis=-1):
         splits = np.split(self.data, num_splits, axis=axis)

         out = [Value(s, (self,), f'split_{i}') for i, s in enumerate(splits)]

         for i, o in enumerate(out):

          def make_backward(idx, child):
            def _backward():
                grad_contrib = np.zeros_like(self.data)
                chunk_size = self.data.shape[axis] // num_splits
                start = idx * chunk_size
                end = (idx + 1) * chunk_size


                slices = [slice(None)] * self.data.ndim
                slices[axis] = slice(start, end)


                grad_contrib[tuple(slices)] = child.grad
                self.grad += grad_contrib
            return _backward

          o._backward = make_backward(i, o)

         return out
    def concat(values, axis=1):
        data = np.concatenate([v.data for v in values], axis=axis)
        out = Value(data, values, 'concat')

        def _backward():
          grads = np.split(out.grad, len(values), axis=axis)
          for v, g in zip(values, grads):
            v.grad += g

        out._backward = _backward
        return out

    def backward(self, clip_value=1.0):
        topo = []
        visited = set()

        def build_topo(a):
            if a not in visited:
                visited.add(a)
                for child in a._prev:
                    build_topo(child)
                topo.append(a)

        build_topo(self)

        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()



# x = Value(np.array([[2.0, 3.0],[4.0, 5.0]]))
# y = Value(np.array([[3.0,9.0],[6.0,8.0]]))
# a=Value(np.array([[1.0,2.0,3.0],[9.0,8.0,7.0]]))
# b=Value(np.array([[2.0,3.0],[4.0,5.0],[6.0,7.0]]))
# z =x+y+x+y+a@b

# z.backward()



# print(a.grad)
# print(b.grad)
# x = Value(np.array([[2.0, 3.0],
#                     [4.0, 5.0]]))

# z = (x ) ** 2
# z.backward()

# print("x.grad:\n", x.grad)
# x = Value(np.array([[100.0, 200.0, 300.0]]))

# y = x.softmax()


# y.backward()

# print("x.grad:\n", x.grad)
def xavier_init(n_in, n_out):
    # Formula: sqrt(6 / (n_in + n_out))
    limit = np.sqrt(6 / (n_in + n_out))


    weights = np.random.uniform(-limit, limit, size=(n_in, n_out))


    return Value(weights)
def normalize_mfcc(mfcc_data):

    mean = np.mean(mfcc_data, axis=1, keepdims=True) # Shape: (16, 1, 13)


    std = np.std(mfcc_data, axis=1, keepdims=True)   # Shape: (16, 1, 13)

    normalized_mfcc = (mfcc_data - mean) / (std + 1e-8)

    return normalized_mfcc


class encoder:
    def __init__(self,d_model):
        self.d_model=d_model
        self.w_d_v=xavier_init(d_model, d_model)
        self.w_d_k=xavier_init(d_model, d_model)
        # self.voice_embedding=Value((voice_embedding))
        self.w1=xavier_init(13, d_model)
        self.wq=xavier_init(d_model, d_model)
        self.wk=xavier_init(d_model, d_model)
        self.wv=xavier_init(d_model, d_model)
        self.gama1_1=Value(np.ones(d_model))
        self.beta1_1=Value(np.zeros(d_model))
        self.gama1_2=Value(np.ones(d_model))
        self.beta1_2=Value(np.zeros(d_model))
        self.gama1_3=Value(np.ones(d_model))
        self.beta1_3=Value(np.zeros(d_model))
        self.gama1_4=Value(np.ones(d_model))
        self.beta1_4=Value(np.zeros(d_model))
        self.gama1_5=Value(np.ones(d_model))
        self.beta1_5=Value(np.zeros(d_model))
        self.gama1_6=Value(np.ones(d_model))
        self.beta1_6=Value(np.zeros(d_model))
        self.gama1_7=Value(np.ones(d_model))
        self.beta1_7=Value(np.zeros(d_model))
        self.gama1_8=Value(np.ones(d_model))
        self.beta1_8=Value(np.zeros(d_model))
        self.gama1_9=Value(np.ones(d_model))
        self.beta1_9=Value(np.zeros(d_model))
        self.gama1_10=Value(np.ones(d_model))
        self.beta1_10=Value(np.zeros(d_model))
        self.gama1_11=Value(np.ones(d_model))
        self.beta1_11=Value(np.zeros(d_model))
        self.gama1_12=Value(np.ones(d_model))
        self.beta1_12=Value(np.zeros(d_model))


        self.ffnW1=xavier_init(d_model, d_model*4)
        self.ffnW2=xavier_init(d_model*4, d_model)
        self.ffnbiasd1=Value(np.zeros(d_model * 4))
        self.ffnbiasd2=Value(np.zeros(d_model ))
    def changedim(self,embedding):
        print("embedding shape",embedding.data.shape)
        print("w1 shape",self.w1.data.shape)
        result=embedding@self.w1
        return result
    def postionalencoding(self,embedding):
        print("embedding shape in positional encoding",embedding.data.shape)

        batch_size=embedding.data.shape[0]
        seq_len=embedding.data.shape[1]
        d_model=embedding.data.shape[2]

        pe=np.zeros((batch_size,seq_len,d_model))
        print(type(embedding))
        print("pe type",type(pe))
        print("pe shape",pe.shape)
        for pos in range(seq_len):
            for i in range(d_model):
                if i%2==0:
                    pe[:,pos,i]=np.sin(pos/(10000**(2 * i / d_model)))
                else:
                    pe[:,pos,i]=np.cos(pos/(10000**(2 * i / d_model)))

        return embedding + Value(pe)
    def softmax(self,x):
        exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def multiheadattaion(self,embedding,num_head=4):


        q=(embedding @ self.wq).split(num_head, axis=2)
        k=(embedding @ self.wk).split(num_head, axis=2)
        v=(embedding @ self.wv).split(num_head, axis=2)
        head_outputs=[]
        for i in range(num_head):
            scores=q[i] @ k[i].T()
            scores=scores / np.sqrt(self.d_model/num_head)
            # print('encoder attention score',np.min(scores.data),np.max(scores.data))
            attn_weights=scores.softmax()
            head_output=attn_weights @ v[i]
            head_outputs.append(head_output)
        return Value.concat(head_outputs, axis=2)
    def FFn(self,x):

       hidden = x @ self.ffnW1 + self.ffnbiasd1
       hidden = hidden.relu()
       out1 = hidden @ self.ffnW2 + self.ffnbiasd2
       return out1

# print("gama1.grad:\n", enc.gama1.grad)
# print("beta1.grad:\n", enc.beta1.grad)
# print("gama2.grad:\n", enc.gama2.grad)
# print("beta2.grad:\n", enc.beta2.grad)
# print("shape of gama1.grad:", enc.gama1.grad.shape)
# print("shape of beta1.grad:", enc.beta1.grad.shape)
# print("shape of gama2.grad:", enc.gama2.grad.shape)
# print("shape of beta2.grad:", enc.beta2.grad.shape)
# print("min and max of gama1.grad:", np.min(enc.gama1.grad), np.max(enc.gama1.grad))
# print("min and max of beta1.grad:", np.min(enc.beta1.grad), np.max(enc.beta1.grad))
# print("min and max of gama2.grad:", np.min(enc.gama2.grad), np.max(enc.gama2.grad))
# print("min and max of beta2.grad:", np.min(enc.beta2.grad), np.max(enc.beta2.grad))
# print("wq,wv,wk min and max:", np.min(enc.wq.grad), np.max(enc.wq.grad), np.min(enc.wv.grad), np.max(enc.wv.grad), np.min(enc.wk.grad), np.max(enc.wk.grad))
class decoder:
    def __init__(self,voacb_size,d_model):
        self.vocab_size=voacb_size
        self.d_model=d_model
        self.embedding = xavier_init(voacb_size, d_model)
        self.w_mask_q=xavier_init(d_model, d_model)
        self.w_mask_k=xavier_init(d_model, d_model)
        self.w_mask_v=xavier_init(d_model, d_model)
        self.w_q=xavier_init(d_model, d_model)
        self.ffnw1=xavier_init(d_model, d_model*4)
        self.ffnw2=xavier_init(d_model*4, d_model)
        self.ffnbias1 = Value(np.zeros(d_model * 4))
        self.ffnbias2 = Value(np.zeros(d_model))
        self.gama2_1=Value(np.ones(d_model))
        self.beta2_1=Value(np.zeros(d_model))
        self.gama2_2=Value(np.ones(d_model))
        self.beta2_2=Value(np.zeros(d_model))
        self.gama2_3=Value(np.ones(d_model))
        self.beta2_3=Value(np.zeros(d_model))
        self.gama2_4=Value(np.ones(d_model))
        self.beta2_4=Value(np.zeros(d_model))
        self.gama2_5=Value(np.ones(d_model))
        self.beta2_5=Value(np.zeros(d_model))
        self.gama2_6=Value(np.ones(d_model))
        self.beta2_6=Value(np.zeros(d_model))
        self.gama2_7=Value(np.ones(d_model))
        self.beta2_7=Value(np.zeros(d_model))
        self.gama2_8=Value(np.ones(d_model))
        self.beta2_8=Value(np.zeros(d_model))
        self.gama2_9=Value(np.ones(d_model))
        self.beta2_9=Value(np.zeros(d_model))
        self.gama2_10=Value(np.ones(d_model))
        self.beta2_10=Value(np.zeros(d_model))
        self.gama2_11=Value(np.ones(d_model))
        self.beta2_11=Value(np.zeros(d_model))
        self.gama2_12=Value(np.ones(d_model))
        self.beta2_12=Value(np.zeros(d_model))
        self.gama2_13=Value(np.ones(d_model))
        self.beta2_13=Value(np.zeros(d_model))
        self.gama2_14=Value(np.ones(d_model))
        self.beta2_14=Value(np.zeros(d_model))
        self.gama2_15=Value(np.ones(d_model))
        self.beta2_15=Value(np.zeros(d_model))
        self.gama2_16=Value(np.ones(d_model))
        self.beta2_16=Value(np.zeros(d_model))
        self.gama2_17=Value(np.ones(d_model))
        self.beta2_17=Value(np.zeros(d_model))
        self.gama2_18=Value(np.ones(d_model))
        self.beta2_18=Value(np.zeros(d_model))
        self.w_d_q=xavier_init(d_model, d_model)
        self.w_d_k=xavier_init(d_model, d_model)
        self.w_d_v=xavier_init(d_model, d_model)
        self.out_proj = xavier_init(d_model, self.vocab_size)
    def embed(self, indices):
        emb = self.embedding.data[indices]
        return Value(emb)
    def decoderpostionalencoding(self,embdding):
        batch_size=embdding.data.shape[0]
        seq_len=embdding.data.shape[1]
        d_model=embdding.data.shape[2]
        pe=np.zeros((batch_size,seq_len,d_model))
        for pos in range(seq_len):
            for i in range(d_model):
                if i%2==0:
                    pe[:,pos,i]=np.sin(pos/(10000**(2 * i / d_model)))
                else:
                    pe[:,pos,i]=np.cos(pos/(10000**(2 * i / d_model)))

        return embdding + Value(pe)
    def maskedmultiheadaattention(self,embedding,num_head=4):
        q=(embedding@self.w_mask_q).split(num_head, axis=2)
        k=(embedding@self.w_mask_k).split(num_head, axis=2)
        v=(embedding@self.w_mask_v).split(num_head, axis=2)
        head_outputs=[]
        for i in range(num_head):
            sores=q[i]@k[i].T()
            sores=sores/np.sqrt(self.d_model/num_head)
            # Apply causal mask
            seq_len = sores.data.shape[-1]
            mask = np.tril(np.ones((seq_len, seq_len)))
            sores = sores.masked_fill(Value(mask == 0), float('-inf'))
            attem_weights=sores.softmax()
            head_output=attem_weights@v[i]
            head_outputs.append(head_output)
        return Value.concat(head_outputs, axis=2)
    def crossmultiheadaattention(self,enocder_k,encoder_v,queery,num_head=4):
        k=(enocder_k@self.w_d_k).split(num_head,axis=2)
        v=(encoder_v@self.w_d_v).split(num_head,axis=2)
        q=(queery@self.w_d_q).split(num_head,axis=2)
        head_output=[]
        for i in range(num_head):
            score=q[i]@k[i].T()

            scores=score/(np.sqrt(self.d_model/num_head))
            # print("score crossmultihead",np.min(scores.data),np.max(scores.data))
            # print('score of cross attention',scores.data)
            atten_weights=scores.softmax()
            head_output1=atten_weights@v[i]
            head_output.append(head_output1)
        return Value.concat(head_output, axis=2)
    def FFn(self,x):
        # x: (5, 117, 128), ffnw1: (128, 512), bias1: (512,)
       hidden = x @ self.ffnw1 + self.ffnbias1
       hidden = hidden.relu()
       out1 = hidden @ self.ffnw2 + self.ffnbias2
       return out1


def audio_mfcc(aduio_file,sr,no_of_frame):
    import wave
    audio=wave.open(aduio_file)
    sr=audio.getframerate()
    num_sample=audio.getnframes()
    audio_data=audio.readframes(num_sample)
    signal=np.frombuffer(audio_data, dtype=np.int16).copy()
    signal=signal.astype(np.float32)
    signal=np.append(signal[0],signal[1:]-0.97*signal[:-1])
    frame_size = int(len(signal) / no_of_frame * 2)   # total_length = (num_frames - 1) * hop_length + frame_size
    hop_length = int(frame_size // 2)
    padding_length=no_of_frame * hop_length + (frame_size - hop_length)
    padded_signal=np.append(signal,np.zeros(padding_length))
    indcies=np.tile(np.arange(0,frame_size),(no_of_frame,1))+np.tile(np.arange(0,no_of_frame*hop_length,hop_length),(frame_size,1)).T
    frames=padded_signal[indcies.astype(np.int32,copy=False)]

    frames*=np.hanning(frame_size)
    signal=np.fft.rfft(frames,512)
    power_spectrum=(1.0/512)*np.square(np.abs(signal))
    low_freq_mel=0
    high_freq_mel=2595*np.log10(1+(sr/2)/700)
    mel_points=np.linspace(low_freq_mel,high_freq_mel,13+2)
    hz_points=700*(10**(mel_points/2595)-1)
    bin=np.floor((512+1)*hz_points/sr)
    num_filter=13

    from librosa.filters import mel
    filter_bank=mel(sr=sr,n_fft=512,n_mels=num_filter,fmin=0,fmax=sr/2)
    #S(k)

    filter_banks=np.transpose(np.dot(power_spectrum,filter_bank.T))
    #apply DCT
    from scipy.fftpack import dct
    mfccs=dct(filter_banks,type=2,axis=1,norm='ortho')
    mfccs=np.transpose(mfccs)


    return mfccs
def parse_encoded(x):
    if isinstance(x, (list, np.ndarray)):
        return np.array(x)

    nums = re.findall(r'\d+', str(x))
    return np.array(list(map(int, nums)))


def data_loader(audio_folder, csv_path, batch_size=5):
    df = pd.read_csv(csv_path)

    df["encoded"] = df["encoded"].apply(parse_encoded)

    id_to_encoded = dict(zip(df["id"].astype(str), df["encoded"]))

    audio_paths = []
    encoded_batch = []


    all_files = []
    for file in os.listdir(audio_folder):
        if not file.endswith(".wav"):
            continue
        audio_id = os.path.splitext(file)[0]
        if audio_id not in id_to_encoded:
            continue
        all_files.append(file)


    all_files.sort()

    print(f"Total audio files found: {len(all_files)}")
    print(f"Total unique IDs in CSV: {len(id_to_encoded)}")

    loaded_ids = set()

    for file in all_files:
        audio_id = os.path.splitext(file)[0]

        if audio_id in loaded_ids:
            print(f"WARNING: Duplicate ID found: {audio_id}")
        loaded_ids.add(audio_id)

        full_path = os.path.join(audio_folder, file)

        audio_paths.append(full_path)
        encoded_batch.append(id_to_encoded[audio_id])

        if len(audio_paths) == batch_size:
            yield audio_paths, encoded_batch
            audio_paths = []
            encoded_batch = []
    if audio_paths and len(audio_paths) == batch_size:
        yield audio_paths, encoded_batch

    print(f"Total unique files loaded: {len(loaded_ids)}")

def collect_parameters(enc, dec):
    params = []


    params.extend([
        enc.w1,
        enc.wq, enc.wk, enc.wv,
        enc.w_d_v, enc.w_d_k,
        enc.ffnW1, enc.ffnW2,
        enc.ffnbiasd1, enc.ffnbiasd2
    ])


    for i in range(1, 13):
        params.append(getattr(enc, f'gama1_{i}'))
        params.append(getattr(enc, f'beta1_{i}'))


    params.extend([
        dec.embedding,
        dec.w_mask_q, dec.w_mask_k, dec.w_mask_v,
        dec.w_q,
        dec.w_d_q, dec.w_d_k, dec.w_d_v,
        dec.ffnw1, dec.ffnw2,
        dec.ffnbias1, dec.ffnbias2,
        dec.out_proj
    ])


    for i in range(1, 19):
        params.append(getattr(dec, f'gama2_{i}'))
        params.append(getattr(dec, f'beta2_{i}'))

    return params


def tarning(epoch, batch_size, lr):

    with open(r"/kaggle/input/datasets/krrishsharma008/voicexyz1/vocab.pkl", "rb") as f:
        vocab_dict = pickle.load(f)

        vocab_size = len(vocab_dict)
    print(f"Loaded vocab with {vocab_size} tokens")
    if "<s>" not in vocab_dict:
      vocab_dict["<s>"] = len(vocab_dict)

    if "<e>" not in vocab_dict:
       vocab_dict["<e>"] = len(vocab_dict)

    print(vocab_dict["<s>"], vocab_dict["<e>"])
    vocab_size = len(vocab_dict)
    print("final len of vocab",len(vocab_dict))
    vocab_size = len(vocab_dict)


    checkpoint_dir = os.getcwd()

    best_loss = float('inf')
    no_improve_epochs = 0
    patience = 5

    # Batch-level learning rate reduction
    best_batch_loss = float('inf')
    no_improve_batches = 0
    batch_patience = 500

    # Global best loss tracking for every 50 batches
    best_loss_global = float('inf')
    global_batch_count = 0

    enc = encoder(128)
    dec = decoder(vocab_size, 128)

    params = collect_parameters(enc, dec)


    optimizer = Adam(params, lr=lr)

    for ep in range(epoch):
        print(f" EPOCH {ep+1}/{epoch} ")


        loader = data_loader(
            audio_folder=r"/kaggle/input/datasets/krrishsharma008/voice123/train/audio",
            csv_path=r"/kaggle/input/datasets/krrishsharma008/voicexyz/final_text2.csv",
            batch_size=batch_size
        )

        batch_num = 0
        total_loss = 0


        for audio_paths, encoded_texts in loader:
            batch_num += 1
            global_batch_count += 1


            mfcc_batch = []
            for audio_file in audio_paths:
                mfcc = audio_mfcc(audio_file, sr=22400, no_of_frame=128)
                mfcc_batch.append(mfcc)

            mfcc_batch = np.stack(mfcc_batch, axis=0)
            encoded_text_batch = np.array(encoded_texts)
            batch_size, seq_len = encoded_text_batch.shape

            # Teacher forcing: decoder input is [<s>, w1, w2, ..., w(seq_len-1)]
            # target is [w1, w2, ..., w(seq_len-1), <e>]
            start_token = vocab_dict["<s>"]
            end_token = vocab_dict["<e>"]

            decoder_input = np.zeros((batch_size, seq_len + 1), dtype=np.int64)
            target1 = np.zeros((batch_size, seq_len + 1), dtype=np.int64)
            decoder_input[:, 0] = start_token
            decoder_input[:, 1:] = encoded_text_batch

            target1[:, :-1] = encoded_text_batch
            target1[:, -1] = end_token

            print(f"Batch {batch_num}: Audio {mfcc_batch.shape}, Text {encoded_text_batch.shape}, Decoder input {decoder_input.shape}, Target {target1.shape}")
            print(f"  Files: {[os.path.basename(p) for p in audio_paths[:2]]}...")  # Show first 2 files


            mfcc_batch=normalize_mfcc(mfcc_batch)
            x = Value(mfcc_batch)
            x1 = enc.changedim(x)
            x2 = enc.postionalencoding(x1)

            x3 = enc.multiheadattaion(x2)
            x4 = x2 + x3
            x5 = x4.layernorm(x4, enc.gama1_1, enc.beta1_1)
            x6 = enc.FFn(x5)
            x7 = x5 + x6
            x8 = x7.layernorm(x7, enc.gama1_2, enc.beta1_2)

            x9 = enc.multiheadattaion(x8)
            x10 = x8 + x9
            x11 = x10.layernorm(x10, enc.gama1_3, enc.beta1_3)
            x12 = enc.FFn(x11)
            x13 = x11 + x12
            x14 = x13.layernorm(x13, enc.gama1_4, enc.beta1_4)

            x15 = enc.multiheadattaion(x14)
            x16 = x14 + x15
            x17 = x16.layernorm(x16, enc.gama1_5, enc.beta1_5)
            x18 = enc.FFn(x17)
            x19 = x17 + x18
            x20 = x19.layernorm(x19, enc.gama1_6, enc.beta1_6)

            x21 = enc.multiheadattaion(x20)
            x22 = x20 + x21
            x23 = x22.layernorm(x22, enc.gama1_7, enc.beta1_7)
            x24 = enc.FFn(x23)
            x25 = x23 + x24
            x26 = x25.layernorm(x25, enc.gama1_8, enc.beta1_8)

            x27 = enc.multiheadattaion(x26)
            x28 = x26 + x27
            x29 = x28.layernorm(x28, enc.gama1_9, enc.beta1_9)
            x30 = enc.FFn(x29)
            x31 = x29 + x30
            x32 = x31.layernorm(x31, enc.gama1_10, enc.beta1_10)

            x33 = enc.multiheadattaion(x32)
            x34 = x32 + x33
            x35 = x34.layernorm(x34, enc.gama1_11, enc.beta1_11)
            x36 = enc.FFn(x35)
            x37 = x35 + x36
            x38 = x37.layernorm(x37, enc.gama1_12, enc.beta1_12)


            d1 = dec.embedding.get_embedding(decoder_input)
            d2 = dec.decoderpostionalencoding(d1)

            d3 = dec.maskedmultiheadaattention(d2)
            d4 = d2 + d3
            d5 = d4.layernorm(d4, dec.gama2_1, dec.beta2_1)
            d6 = dec.crossmultiheadaattention(x14, x14, d5)
            d7 = d5 + d6
            d8 = d7.layernorm(d7, dec.gama2_2, dec.beta2_2)
            d9 = dec.FFn(d8)
            d10 = d8 + d9
            d11 = d10.layernorm(d10,dec.gama2_3, dec.beta2_3)

            d12 = dec.maskedmultiheadaattention(d11)
            d13 = d11 + d12
            d14 = d13.layernorm(d13,dec.gama2_4, dec.beta2_4)
            d15 = dec.crossmultiheadaattention(x14, x14, d14)
            d16 = d14 + d15
            d17 = d16.layernorm(d16, dec.gama2_5, dec.beta2_5)
            d18 = dec.FFn(d17)
            d19 = d17 + d18
            d20 = d19.layernorm(d19, dec.gama2_6, dec.beta2_6)

            d21 = dec.maskedmultiheadaattention(d20)
            d22 = d20 + d21
            d23 = d22.layernorm(d22, dec.gama2_7, dec.beta2_7)
            d24 = dec.crossmultiheadaattention(x38, x38, d23)
            d25 = d23 + d24
            d26 = d25.layernorm(d25, dec.gama2_8, dec.beta2_8)
            d27 = dec.FFn(d26)
            d28 = d26 + d27
            d29 = d28.layernorm(d28, dec.gama2_9, dec.beta2_9)

            d30 = dec.maskedmultiheadaattention(d29)
            d31 = d29 + d30
            d32 = d31.layernorm(d31, dec.gama2_10, dec.beta2_10)
            d33 = dec.crossmultiheadaattention(x38, x38, d32)
            d34 = d32 + d33
            d35 = d34.layernorm(d34, dec.gama2_11, dec.beta2_11)
            d36 = dec.FFn(d35)
            d37 = d35 + d36
            d38 = d37.layernorm(d37, dec.gama2_12, dec.beta2_12)

            d39 = dec.maskedmultiheadaattention(d38)
            d40 = d38 + d39
            d41 = d40.layernorm(d40, dec.gama2_13, dec.beta2_13)
            d42 = dec.crossmultiheadaattention(x38, x38, d41)
            d43 = d41 + d42
            d44 = d43.layernorm(d43, dec.gama2_14, dec.beta2_14)
            d45 = dec.FFn(d44)
            d46 = d44 + d45
            d47 = d46.layernorm(d46,dec.gama2_15, dec.beta2_15)

            d48 = dec.maskedmultiheadaattention(d47)
            d49 = d47 + d48
            d50 = d49.layernorm(d49, dec.gama2_16, dec.beta2_16)
            d51 = dec.crossmultiheadaattention(x38, x38, d50)
            d52 = d50 + d51
            d53 = d52.layernorm(d52, dec.gama2_17, dec.beta2_17)
            d54 = dec.FFn(d53)
            d55 = d53 + d54
            d56 = d55.layernorm(d55, dec.gama2_18, dec.beta2_18)
            print(d56.data.shape)


            logits = d11 @ dec.out_proj  # (batch, seq_len+1, vocab_size)



            batch_size_tf, seq_len_tf = target1.shape
            target_arr = np.zeros((batch_size_tf, seq_len_tf, vocab_size), dtype=np.float64)
            unk_token = vocab_dict.get("<unk>", 1)
            for i in range(batch_size_tf):
                for j in range(seq_len_tf):
                    tok = int(target1[i, j])
                    if tok < 0 or tok >= vocab_size:
                        tok = unk_token
                    target_arr[i, j, tok] = 1.0
            weight_array = np.ones((batch_size_tf, seq_len_tf), dtype=np.float64)
            weight_array[target1 == 2] = 0.1
            target = Value(target_arr)
            raw_loss = logits.crossentropy_with_logits(target, label_smoothing=0.0)
            mask_array = (target1 != 0).astype(np.float64)
            mask = Value(mask_array)
            weight_value = Value(weight_array)
            masked_loss = raw_loss * mask * weight_value

            masked_loss = raw_loss * mask*weight_value


            valid_tokens = float((mask_array * weight_array).sum())

            loss=masked_loss.sum() / valid_tokens
            current_loss_val = float(loss.data)
            print(f"  Loss: {loss.data:.6f}")
            total_loss += loss.data
            loss.backward()



            # print("Max Gradient1:", np.max(np.abs(dec.out_proj.grad)))

            # Update parameters using Adam optimizer
            if global_batch_count < 1000:
                 optimizer.lr = lr * (global_batch_count / 1000)
            else:
                 optimizer.lr = lr
            optimizer.step()

            if global_batch_count % 5 == 0:

                # axis=-1 par argmax lene se har position ka best token milega
                preds = np.argmax(logits.data[0], axis=-1)

                # target1[0] batch ka pehla asli target hai
                actual_target = target1[0].astype(int)
                preds = np.argmax(logits.data[0], axis=-1)
                non_space = [(i, p) for i, p in enumerate(preds[:20]) if p != 2]
                print("Non-space predictions:", non_space[:10])


                print(f"Target (Pehle 20 tokens):    {actual_target[:20]}")
                print(f"Predicted (Pehle 20 tokens): {preds[:20]}")
                print("="*50 + "\n")


            # Update parameters using Adam optimizer

            import gc

            # Zero gradients for next batch
            optimizer.zero_grad()
            del x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38
            del d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31, d32, d33, d34, d35, d36, d37, d38, d39, d40, d41, d42, d43, d44, d45, d46, d47, d48, d49, d50, d51, d52, d53, d54, d55, d56


            del logits,  target_arr, target, raw_loss, mask_array, mask, masked_loss, loss


            del mfcc_batch, encoded_text_batch, decoder_input, target1

            gc.collect()


            if current_loss_val < best_batch_loss:
                best_batch_loss = current_loss_val
                no_improve_batches = 0
            else:
                no_improve_batches += 1

            if no_improve_batches >= batch_patience:
                optimizer.lr *= 0.2
                print(f"Reducing learning rate to {optimizer.lr} after {batch_patience} batches without improvement")
                no_improve_batches = 0
                best_batch_loss = float('inf')

            # Save best model every 500 batches
            if global_batch_count % 500 == 0:
                if current_loss_val < best_loss_global:
                    best_loss_global = current_loss_val
                    best_model_ckpt = os.path.join(checkpoint_dir, f"best_model_batch_{global_batch_count:05d}_loss_{current_loss_val:.6f}.pkl")
                    named_params = {
                   'enc_w1': enc.w1.data,
                   'enc_wq': enc.wq.data,
                   'enc_wk': enc.wk.data,
                   'enc_wv': enc.wv.data,
                   'enc_w_d_v': enc.w_d_v.data,
                   'enc_w_d_k': enc.w_d_k.data,
                   'enc_ffnW1': enc.ffnW1.data,
                   'enc_ffnW2': enc.ffnW2.data,
                   'enc_ffnbias1': enc.ffnbiasd1.data,
                   'enc_ffnbias2': enc.ffnbiasd2.data,
                     }
                    for i in range(1, 13):
                       named_params[f'enc_gama1_{i}'] = getattr(enc, f'gama1_{i}').data
                       named_params[f'enc_beta1_{i}'] = getattr(enc, f'beta1_{i}').data

                    named_params.update({
                  'dec_embedding': dec.embedding.data,
                  'dec_w_mask_q': dec.w_mask_q.data,
                  'dec_w_mask_k': dec.w_mask_k.data,
                  'dec_w_mask_v': dec.w_mask_v.data,
                  'dec_w_q': dec.w_q.data,
                  'dec_w_d_q': dec.w_d_q.data,
                  'dec_w_d_k': dec.w_d_k.data,
                  'dec_w_d_v': dec.w_d_v.data,
                  'dec_ffnw1': dec.ffnw1.data,
                  'dec_ffnw2': dec.ffnw2.data,
                  'dec_ffnbias1': dec.ffnbias1.data,
                  'dec_ffnbias2': dec.ffnbias2.data,
                  'dec_out_proj': dec.out_proj.data,
                   })
                    for i in range(1, 19):
                      named_params[f'dec_gama2_{i}'] = getattr(dec, f'gama2_{i}').data
                      named_params[f'dec_beta2_{i}'] = getattr(dec, f'beta2_{i}').data
                    with open(best_model_ckpt, 'wb') as f:
                        pickle.dump(named_params, f)
                    print(f"✓ Best model saved at batch {global_batch_count}: {best_model_ckpt}")







        avg_loss = total_loss / batch_num
        print(f"Epoch {ep+1} complete. Avg Loss: {avg_loss:.6f}\n")


        epoch_ckpt = os.path.join(checkpoint_dir, f"epoch_{ep+1:03d}_loss_{avg_loss:.6f}.pkl")
        named_params = {
                   'enc_w1': enc.w1.data,
                   'enc_wq': enc.wq.data,
                   'enc_wk': enc.wk.data,
                   'enc_wv': enc.wv.data,
                   'enc_w_d_v': enc.w_d_v.data,
                   'enc_w_d_k': enc.w_d_k.data,
                   'enc_ffnW1': enc.ffnW1.data,
                   'enc_ffnW2': enc.ffnW2.data,
                   'enc_ffnbias1': enc.ffnbiasd1.data,
                   'enc_ffnbias2': enc.ffnbiasd2.data,
                     }
        for i in range(1, 13):
                       named_params[f'enc_gama1_{i}'] = getattr(enc, f'gama1_{i}').data
                       named_params[f'enc_beta1_{i}'] = getattr(enc, f'beta1_{i}').data

        named_params.update({
                  'dec_embedding': dec.embedding.data,
                  'dec_w_mask_q': dec.w_mask_q.data,
                  'dec_w_mask_k': dec.w_mask_k.data,
                  'dec_w_mask_v': dec.w_mask_v.data,
                  'dec_w_q': dec.w_q.data,
                  'dec_w_d_q': dec.w_d_q.data,
                  'dec_w_d_k': dec.w_d_k.data,
                  'dec_w_d_v': dec.w_d_v.data,
                  'dec_ffnw1': dec.ffnw1.data,
                  'dec_ffnw2': dec.ffnw2.data,
                  'dec_ffnbias1': dec.ffnbias1.data,
                  'dec_ffnbias2': dec.ffnbias2.data,
                  'dec_out_proj': dec.out_proj.data,
                   })
        for i in range(1, 19):
                      named_params[f'dec_gama2_{i}'] = getattr(dec, f'gama2_{i}').data
                      named_params[f'dec_beta2_{i}'] = getattr(dec, f'beta2_{i}').data
        with open(epoch_ckpt, 'wb') as f:
                      pickle.dump(named_params, f)
        print(f"Saved epoch checkpoint: {epoch_ckpt}")


        if avg_loss < best_loss:
            print("Loss decreased this epoch; per request do not save best checkpoint on decrease.")
            best_loss = avg_loss
            no_improve_epochs = 0
            improved = True
        else:
            no_improve_epochs += 1
            improved = False
            print(f"No improvement streak: {no_improve_epochs}/{patience}")


        if not improved:
            nonimp_ckpt = os.path.join(checkpoint_dir, "nonimprovement_latest.pkl")
            with open(nonimp_ckpt, 'wb') as f:
                pickle.dump(named_params, f)
            print(f"Saved non-improvement checkpoint: {nonimp_ckpt}")
        else:
            print("Skipped non-improvement checkpoint because loss decreased.")

        # Append loss to CSV (able to inspect later)



        if no_improve_epochs >= patience:
            print(f"Early stopping triggered: no improvement over {patience} epochs.")
            break
# def train_single_sample(audio_path, target_text, epochs=200, lr=0.0001):
#     # 1. Preprocessing (Single Sample)
#     # MFCC shape: (1, 128, n_mfcc)
#     mfcc = audio_mfcc(audio_path, sr=22400, no_of_frame=128)
#     mfcc_batch = np.expand_dims(normalize_mfcc(mfcc), axis=0)
#     with open(r"/kaggle/input/datasets/krrishsharma008/voicexyz1/vocab.pkl", "rb") as f:
#         vocab_dict = pickle.load(f)
#       # Should be 72
#         vocab_size = len(vocab_dict)
#     print(f"Loaded vocab with {vocab_size} tokens")
#     if "<s>" not in vocab_dict:
#       vocab_dict["<s>"] = len(vocab_dict)

#     if "<e>" not in vocab_dict:
#        vocab_dict["<e>"] = len(vocab_dict)

#     print(vocab_dict["<s>"], vocab_dict["<e>"])
#     vocab_size = len(vocab_dict)
#     print("final len of vocab",len(vocab_dict))
#     vocab_size = len(vocab_dict)

#     # Text Encoding
#     target_text = target_text.strip()

# # 2. Encoding with fallback to <unk> (index 1)
#     encoded_text = []
#     for char in target_text:
#       idx = vocab_dict.get(char, 1) # Agar char nahi mila toh 1
#       encoded_text.append(idx)


#     batch_size, seq_len = np.array([encoded_text]).shape
#     start_token = vocab_dict["<s>"]
#     end_token = vocab_dict["<e>"]

#     # Decoder Input & Target setup
#     decoder_input_np = np.zeros((1, seq_len + 1), dtype=np.int64)
#     target_np = np.zeros((1, seq_len + 1), dtype=np.int64)
#     decoder_input_np[:, 0] = start_token
#     decoder_input_np[:, 1:] = encoded_text
#     target_np[:, :-1] = encoded_text
#     target_np[:, -1] = end_token

#     # Padding Mask for this specific sample (1 jahan data hai, 0 jahan padding)
#     # Kyunki ye single sample hai, agar sequence length fixed hai toh ye sab 1 honge
#     tgt_pad_mask = (decoder_input_np != 0).astype(np.float64)
#     enc = encoder(64)
#     dec = decoder(vocab_size, 64)
#     params = collect_parameters(enc, dec)
#     print(f"Starting Overfit on: {os.path.basename(audio_path)}")
#     optimizer = Adam(params, lr=lr)

#     for ep in range(epochs):
#         optimizer.zero_grad()

#         # --- ENCODER PASS ---
#         x = Value(mfcc_batch)
#         x_enc = enc.changedim(x)
#         x_enc = enc.postionalencoding(x_enc)

#         # Encoder Blocks (Simplified loop for 6 layers)
#         for _ in range(6):
#             attn = enc.multiheadattaion(x_enc)
#             x_enc = (x_enc + attn).layernorm(x_enc + attn, enc.gama1, enc.beta1)
#             ffn = enc.FFn(x_enc)
#             x_enc = (x_enc + ffn).layernorm(x_enc + ffn, enc.gama2, enc.beta2)

#         # --- DECODER PASS ---
#         d = dec.embedding.get_embedding(decoder_input_np)
#         d = dec.decoderpostionalencoding(d)

#         # Decoder Blocks (Passing the mask here!)
#         for _ in range(6):
#             # Self Attention with Causal + Padding Mask
#             d_attn = dec.maskedmultiheadaattention(d)
#             d = (d + d_attn).layernorm(d + d_attn, dec.gama3, dec.beta3)

#             # Cross Attention
#             d_cross = dec.crossmultiheadaattention(x_enc, x_enc, d)
#             d = (d + d_cross).layernorm(d + d_cross, dec.gama4, dec.beta4)

#             # FFN
#             d_ffn = dec.FFn(d)
#             d = (d + d_ffn).layernorm(d + d_ffn, dec.gama3, dec.beta3)

#         # Output & Loss
#         logits = d @ dec.out_proj

#         # Build One-Hot Target
#         target_arr = np.zeros((1, seq_len + 1, vocab_size))
#         for j in range(seq_len + 1):
#             target_arr[0, j, int(target_np[0, j])] = 1.0

#         loss = logits.crossentropy_with_logits(Value(target_arr))

#         # Backward & Step
#         loss.backward()
#         optimizer.step()

#         if ep % 10 == 0:
#             loss = loss.mean()
#             preds = np.argmax(logits.data[0], axis=-1)
#             print(f"Epoch {ep} | Loss: {loss.data.item():.4f}")
#             print(f"Pred: {preds[:15]}")
#             print(f"Targ: {target_np[0][:15]}")

# # Ise run karein
# train_single_sample(audio_path="/kaggle/input/datasets/krrishsharma008/voice123/train/audio/0001_030.wav", target_text="यह है मोटा राजा")


# Run training
tarning(epoch=5, batch_size=16, lr=0.0003)

# After training, zip all model checkpoint .pkl files in current working directory
all_pkl = [f for f in os.listdir(os.getcwd()) if f.endswith('.pkl')]
if all_pkl:
    archive_name = os.path.join(os.getcwd(), 'checkpoints_archive.zip')
    with zipfile.ZipFile(archive_name, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for file_name in all_pkl:
            zf.write(file_name, arcname=file_name)
    print(f"Checkpoints zipped to: {archive_name}")
    print("You can now download checkpoints_archive.zip from Kaggle output.")
else:
    print("No .pkl files found to archive.")

def predict(checkpoint_path, audio_path, vocab_path, max_len=117):

    with open(vocab_path, "rb") as f:
        vocab_dict = pickle.load(f)
    if "<s>" not in vocab_dict:
        vocab_dict["<s>"] = len(vocab_dict)
    if "<e>" not in vocab_dict:
        vocab_dict["<e>"] = len(vocab_dict)
    vocab_size = len(vocab_dict)
    start_token = vocab_dict["<s>"]
    end_token = vocab_dict["<e>"]
    # idx_to_token = {v: k for k, v in vocab_dict.items()}
    idx_to_token = {}
    for k, v in vocab_dict.items():
     try:

        if isinstance(k, int):
            idx_to_token[k] = str(v)

        elif isinstance(v, int):
            idx_to_token[v] = str(k)

        elif str(v).isdigit():
            idx_to_token[int(v)] = str(k)
     except:
        continue



    with open(checkpoint_path, "rb") as f:
        params = pickle.load(f)


    enc = encoder(128)
    dec = decoder(vocab_size, 128)


    enc.w1.data       = params['enc_w1']
    enc.wq.data       = params['enc_wq']
    enc.wk.data       = params['enc_wk']
    enc.wv.data       = params['enc_wv']
    enc.w_d_v.data    = params['enc_w_d_v']
    enc.w_d_k.data    = params['enc_w_d_k']
    enc.ffnW1.data    = params['enc_ffnW1']
    enc.ffnW2.data    = params['enc_ffnW2']
    enc.ffnbiasd1.data = params['enc_ffnbias1']
    enc.ffnbiasd2.data = params['enc_ffnbias2']
    for i in range(1, 13):
        getattr(enc, f'gama1_{i}').data = params[f'enc_gama1_{i}']
        getattr(enc, f'beta1_{i}').data  = params[f'enc_beta1_{i}']


    dec.embedding.data  = params['dec_embedding']
    dec.w_mask_q.data   = params['dec_w_mask_q']
    dec.w_mask_k.data   = params['dec_w_mask_k']
    dec.w_mask_v.data   = params['dec_w_mask_v']
    dec.w_q.data        = params['dec_w_q']
    dec.w_d_q.data      = params['dec_w_d_q']
    dec.w_d_k.data      = params['dec_w_d_k']
    dec.w_d_v.data      = params['dec_w_d_v']
    dec.ffnw1.data      = params['dec_ffnw1']
    dec.ffnw2.data      = params['dec_ffnw2']
    dec.ffnbias1.data   = params['dec_ffnbias1']
    dec.ffnbias2.data   = params['dec_ffnbias2']
    dec.out_proj.data   = params['dec_out_proj']
    for i in range(1, 19):
        getattr(dec, f'gama2_{i}').data = params[f'dec_gama2_{i}']
        getattr(dec, f'beta2_{i}').data  = params[f'dec_beta2_{i}']


    mfcc = audio_mfcc(audio_path, sr=22400, no_of_frame=128)
    mfcc = np.expand_dims(mfcc, axis=0)   # (1, 128, 13)
    def normalize_mfcc(mfcc_data):
        # mfcc_data shape: (16, 32, 13)


        mean = np.mean(mfcc_data, axis=1, keepdims=True) # Shape: (16, 1, 13)


        std = np.std(mfcc_data, axis=1, keepdims=True)   # Shape: (16, 1, 13)


        normalized_mfcc = (mfcc_data - mean) / (std + 1e-8)

        return normalized_mfcc
    mfcc = normalize_mfcc(mfcc)


    x  = Value(mfcc)
    x1 = enc.changedim(x)
    x2 = enc.postionalencoding(x1)

    x3  = enc.multiheadattaion(x2);  x4  = x2 + x3
    x5  = x4.layernorm(x4,  enc.gama1_1, enc.beta1_1)
    x6  = enc.FFn(x5);               x7  = x5 + x6
    x8  = x7.layernorm(x7,  enc.gama1_2, enc.beta1_2)

    x9  = enc.multiheadattaion(x8);  x10 = x8 + x9
    x11 = x10.layernorm(x10, enc.gama1_3, enc.beta1_3)
    x12 = enc.FFn(x11);              x13 = x11 + x12
    x14 = x13.layernorm(x13, enc.gama1_4, enc.beta1_4)

    x15 = enc.multiheadattaion(x14); x16 = x14 + x15
    x17 = x16.layernorm(x16, enc.gama1_5, enc.beta1_5)
    x18 = enc.FFn(x17);              x19 = x17 + x18
    x20 = x19.layernorm(x19, enc.gama1_6, enc.beta1_6)

    x21 = enc.multiheadattaion(x20); x22 = x20 + x21
    x23 = x22.layernorm(x22, enc.gama1_7, enc.beta1_7)
    x24 = enc.FFn(x23);              x25 = x23 + x24
    x26 = x25.layernorm(x25, enc.gama1_8, enc.beta1_8)

    x27 = enc.multiheadattaion(x26); x28 = x26 + x27
    x29 = x28.layernorm(x28, enc.gama1_9,  enc.beta1_9)
    x30 = enc.FFn(x29);              x31 = x29 + x30
    x32 = x31.layernorm(x31, enc.gama1_10, enc.beta1_10)

    x33 = enc.multiheadattaion(x32); x34 = x32 + x33
    x35 = x34.layernorm(x34, enc.gama1_11, enc.beta1_11)
    x36 = enc.FFn(x35);              x37 = x35 + x36
    x38 = x37.layernorm(x37, enc.gama1_12, enc.beta1_12)


    generated = [start_token]

    for _ in range(max_len):
        dec_input = np.array([generated])   # (1, current_len)

        d1  = dec.embedding.get_embedding(dec_input)
        d2  = dec.decoderpostionalencoding(d1)

        d3  = dec.maskedmultiheadaattention(d2); d4  = d2  + d3
        d5  = d4.layernorm(d4,  dec.gama2_1, dec.beta2_1)
        d6  = dec.crossmultiheadaattention(x8,  x8,  d5)
        d7  = d5  + d6
        d8  = d7.layernorm(d7,  dec.gama2_2, dec.beta2_2)
        d9  = dec.FFn(d8);                       d10 = d8  + d9
        d11 = d10.layernorm(d10, dec.gama2_3, dec.beta2_3)

        d12 = dec.maskedmultiheadaattention(d11); d13 = d11 + d12
        d14 = d13.layernorm(d13, dec.gama2_4, dec.beta2_4)
        d15 = dec.crossmultiheadaattention(x14, x14, d14)
        d16 = d14 + d15
        d17 = d16.layernorm(d16, dec.gama2_5, dec.beta2_5)
        d18 = dec.FFn(d17);                       d19 = d17 + d18
        d20 = d19.layernorm(d19, dec.gama2_6, dec.beta2_6)

        last_step_hidden = d11.data[0, -1] # (128,)
        logits = last_step_hidden @ dec.out_proj.data # (74,)

        # Softmax & Argmax
        temperature = 1

        logits = (last_step_hidden @ dec.out_proj.data) / temperature


        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / np.sum(exp_l)


        next_token = int(np.argmax(probs))




        if next_token == end_token:
            break

        generated.append(next_token)


    output_tokens = generated[1:]   
    predicted_text = ''.join([idx_to_token.get(t, '') for t in output_tokens])
    for t in output_tokens:
        print(t)
    print("Predicted:", predicted_text)
    print(dec.w_d_q.data )
    print(dec.w_d_k.data )
    print(dec.w_d_v.data )
    return predicted_text
print(predict(
    checkpoint_path=r"c:\Users\Dell\Downloads\best_model_batch_05000_loss_2.375513.pkl",
    audio_path=r"c:\Users\Dell\Downloads\Hindi_train\train\audio\0001_030.wav",
    vocab_path=r"C:\Users\Dell\OneDrive\Desktop\lmniit\vocab.pkl"
))







