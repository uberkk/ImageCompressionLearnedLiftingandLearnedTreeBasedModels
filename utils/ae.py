import os
import numpy as np
from range_coder import *

class ArithmeticCoder():
    def __init__(self, cdf, fname="tmp.bin", keep_file=False):
        self.cdf = cdf
        self.fname = fname
        self.per_ch_length = 0
        self.keep_file = keep_file
        
    def encode(self, x):
        self.encoder = RangeEncoder(self.fname)
        for i, (d,c) in enumerate( zip(x, self.cdf) ) :
            if len(d):
                try:
                    self.encoder.encode(d.tolist(),c.tolist())
                except Exception as e:
                    print(f'Error in encoding of Ch. {i}')
                    print(np.unique(d))
                    print(c)
                    raise e
        self.encoder.close()
        self.per_ch_length = list(map(len, x))
        total_bits = int(os.stat(self.fname).st_size)*8
        return total_bits

    def decode(self):
        self.decoder = RangeDecoder(self.fname)
        self.decoder_fun = lambda lng, cdf: self.decoder.decode(lng, 
                                                                cdf.tolist())
        stream = list(map(self.decoder_fun, self.per_ch_length, self.cdf))

        self.decoder.close()
        return stream

    def measure_bpp(self, x):
        bits = self.encode(x)
        stream = self.decode()
        for i, (a, b) in enumerate(zip(stream, x)):
            assert (np.all(a == b)), "Ch. {} couldn't be decoded".format(i)
        return bits, stream

    def __del__(self):
        if not(self.keep_file) and os.path.exists(self.fname):
            os.remove(self.fname)


class ContextArithmeticCoderValidator():
    def __init__(self, *args, **kwargs):
        self.coder = ContextArithmeticCoder(*args, **kwargs)
        self.cdf_stack = []
        self.codes_stack = []
        self.org_codes = []
        self.qmin_stack = []

    def encode(self, x, cdf, x_org, qmin):
        self.cdf_stack.append(cdf)
        self.codes_stack.append(x)
        self.org_codes.append(x_org)
        self.qmin_stack.append(qmin)
        self.coder.encode(x, cdf)

    def decode(self):
        stream = []
        for cdf in self.cdf_stack:
            stream.append(self.coder.decode(cdf))

        self.codes_stack = np.array(self.codes_stack)
        self.org_codes = np.array(self.org_codes)
        self.qmin_stack = np.array(self.qmin_stack)
        self.qmin_stack = self.qmin_stack.reshape(self.codes_stack.shape)
        recovered_org = stream + self.qmin_stack
        stream = np.array(stream)
        stream = stream.reshape(self.codes_stack.shape)

        for i, (a, b) in enumerate(zip(stream, self.codes_stack)):
            assert (np.all(a == b)), f"Ch. {i} couldn't be decoded"

        
        for i, (a, b) in enumerate(zip(recovered_org, self.org_codes)):
            err = np.mean(np.abs(a - b))
            assert np.sum(err) == 0, f"Ch. {err} couldn't be decoded"
        return self._total_bits()

    def _total_bits(self):
        return self.coder.total_bits()
        
    def _avg_bits(self):
        return self.coder.total_bits()/self.codes_stack.size


        # for i, (a, b) in enumerate(zip(stream, codes)):
        #     assert (np.all(a == b)), "Ch. {} couldn't be decoded".format(i)

class ContextArithmeticCoder():
    def __init__(self, fname="tmp.bin", keep_file=False):
        self.fname = fname
        self.keep_file = keep_file
        self.encoder = None
        self.decoder = None
        self.mode = None

    def encode(self, x, cdf):
        self.set_mode('encode')
        _ = [self.encoder.encode(d.tolist(),c.tolist())
                for d, c in zip(x, cdf)]

    def set_mode(self, mode):
        assert mode in ['encode', 'decode', 'bits']

        if mode is 'encode':
            if not self.encoder:
                self.encoder = RangeEncoder(self.fname)

        elif mode is 'decode':
            if self.encoder:
                self.encoder.close()
                self.encoder = None
            if not self.decoder:
                self.decoder = RangeDecoder(self.fname)

        elif mode is 'bits':
            if self.encoder:
                self.encoder.close()
                self.encoder = None
            if self.decoder:
                self.decoder.close()
                self.decoder = None

    def total_bits(self):
        self.set_mode('bits')
        total_bits = int(os.stat(self.fname).st_size)*8
        return total_bits

    def decode(self, cdf):
        self.set_mode('decode')
        stream = []
        for c in cdf:
            s = self.decoder.decode(1, c.tolist())
            stream.append(s)
        stream = np.array(stream)
        return stream

    def validate_cdf(self, data, cdf):
        assert len(cdf)>2, 'Size error in CDF!'
        assert cdf[0]==0, 'First element is not zero!'
        assert cdf[-1]==(2**16-1), 'Last element error!'
        assert np.all(np.diff(cdf)>=0), 'Non-monotonic CDF!'
        assert data < len(cdf), 'Codeword out of range!'

    def __del__(self):
        if not(self.keep_file) and os.path.exists(self.fname):
            os.remove(self.fname)