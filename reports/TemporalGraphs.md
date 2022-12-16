# Temporal Graphs
Temporal graphs are graphs where edges have timestamps. 
Here will be description on how to create such graphs and analyze them, as well as the code for such purposes.

* [Data Section](Data.md)

## Empirical notes
* embedding size (out_channels) and feature size (in_channels) should be sufficient big, due to make it possible for scoring matrix to learn
  * scoring: take convolutional inputs, multiply on matrix, multiply on summaries. So the process looks on convolutions, make some basic linear math to compose representation of node within relation type, and try to guess wheter the node was taken part in creation of summary. So if the number of channels will be small, matrix will not be learned
  * my empirical rule is so 
    * power of out channels should be greater than number of nodes
    * power of in channels should be smth like # TODO