import torch as pt

class DistanceEmbedding(pt.nn.Module):

    def __init__( self, *, eps : float = 1e-4 ):
        super().__init__()
        self.eps = eps

    def forward( self, d : pt.Tensor, # Any shape
                ) -> pt.Tensor:
        log_d = pt.log( d + self.eps )
        inv_d = 1.0 / ( d + self.eps )
        return pt.stack( (d, log_d, inv_d), dim=-1 )