Here we presents our own GraphTransFormer, which combines concepts from Graph Neural Networks, attention mechanisms, and MLPs.<br>
<br>
- The model leverages both global attention (to capture the overall graph structure) and local layers (to capture neighborhood relationships).<br>
- This architecture can handle graphs of varying sizes and structures, as long as the number of nodes and the adjacency matrix are provided.<br>
- The use of attention ensures that the model can scale to larger graphs while maintaining computational efficiency.<br>
- All components are differentiable, allowing the model to be trained end-to-end using optimization.<br>
<br>
It's graph, it's very different with a text within a sequential in nature. How we adopt it?<br>
<br>
- Global attention in this architecture can boost LLMs to handling long-range dependencies in text. In LLMs, self-attention typically computes pairwise attention scores between all tokens in a sequence. However, for very long sequence, it's stupid. Now in this architecture, this can summarize the entire sequence into a single vector and use it to guide the token generation. This would allow the model to focus on high-level context while still generating tokens locally.<br>
- Text often containes implicit graph-like structures, such as <b>syntatic dependency trees</b> and <b>knowledge graphs</b>. By representing text as a graph, we could apply this architecture to process these structures. This approach could enhance text generation by incorporating structural information that traditional models might stupid.<br>
- This architecture balances local (neighborhood) and global (attention) information using <i>alpha</i> parameter. Which for local context, the informations captured by standard self-attention over nearby tokens. For global context, the information captured by global summary vector. This 'small' little hybrid approach could improve coherence in long-form text generation by ensuring that both local fluency and global consistency are maintained.<br>
- Text often has hierarchial fucking (sentences -> paragraphs -> documents). Node as sentences or paragraphs, and edges as the relationships like co-reference, discourse relations, or topic similarity (like a RAGs right? No, it's not RAG, buddies)<br>
<br>
<br>
Overall, this architecture is dominated by the most expensive operations<br>
- Embedding Layer :<br>
	<i>O(batch_size * num_nodes * (input_dim * hidden_dim + hidden_dim * embedding_dim))<br></i>
- GNN Layer:<br>
	<i>O(batch_size * num_nodes * num_nodes * embedding_dim + batch_size * num_nodes * embedding_dim * embedding_dim)<br></i>
<br>
<br>
In contrast to another models, which typically quadratic term due to the self-attention, this architecture has a similar quadratic complexity <b>but with respect to sum of the nodes</b>. In conclusion, beware when you set total of nodes. <b>If you blind, your hardware is over</b>
<br>
<br>
<br>
We conducted an architectural evaluation, scaling up the layers of a GPT-2-based model to approximate 10 billion parameters. The results yielded a demonstrably significant improvement, most notably in the model's capacity to differentiate between lexically sophisticated and commonplace language. In the realm of elementary code generation, our model exhibits a notable capacity for maintaining initial informational integrity, particularly within the context of coding tasks not exceeding 10 lines. (It is, however, prudent to note that this model remains idiot than the Qwen 14B coder.) We invite independent replication of this work, encompassing de novo architectural design, pre-training procedures, and deployment. We thank you for your consideration.<br>
<br> <br>
Special thanks to our montir -> Wanto, Tan Hyao, Kristanto, Hang Tou, Yun Han Que
