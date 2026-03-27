# transformer_specch_to_text_use_only_numpy
A pure Python implementation of a Transformer-based Automatic Speech Recognition (ASR) system. Built using a custom Autograd engine (Value-based) to handle backpropagation from scratch. Features a 4-head attention mechanism and encoder-decoder architecture optimized for Hindi phonemes. No high-level deep learning libraries—just raw math and logic.
Why I built this from Scratch?
To be honest, I didn't just want to build another ASR model using torch.nn. I wanted to see under the hood—to understand how a Transformer actually "thinks." By avoiding black-box libraries and building the entire backpropagation engine (the Value class) and attention mechanism from the ground up, I got the chance to really get my hands dirty.

I was curious to see how tweaking individual components—like messing with the Cross-Attention layers or adjusting the Feed-Forward Networks (FFN)—would actually impact the final Hindi output. It wasn't just about getting a result; it was about seeing exactly how the math translates into a spoken word. Building it this way allowed me to observe the "why" and "how" of every single gradient and weight update, which is something you just can't experience when using high-level frameworks.





🛠️ Challenges & Technical Solutions


1. The Manual Backpropagation NightmareUpdating every single weight manually in a Transformer is not just impossible—it’s risky due to the high probability of human error in calculus. To solve this, I implemented a Computational Graph approach. Every weight and operation was treated as a Node, allowing the engine to automatically handle the chain rule and backpropagate gradients through the entire network systematically.
  
  
  
 2. The CPU & Memory Bottleneck (NumPy Constraints)Since I used NumPy, the training was restricted to the CPU, making it significantly slower than GPU-based training. Moreover, the memory footprint was huge, leading to frequent Out of Memory (OOM) errors. While the CPU limitation remained, I optimized memory by manually clearing gradients after every batch and simplifying the architecture to a $d_{model}$ of 128 and 1 Attention Head to keep the RAM usage stable.
    
    3. Explosive Softmax Scores & Weight InitializationInitially, I used naive random weight initialization, which caused the Softmax scores to explode into the range of $10^{12}$. This made the model untrainable. I fixed this by implementing two crucial steps: first, I normalized the MFCC input using $(x - \mu) / \sigma$, and second, I applied Xavier Normalization $\sqrt{6 / (fan\_in + fan\_out)}$ to the weights. This brought the activations back into a stable range.
   
       
      5. Architecture vs. Information LossIn an attempt to save memory, I tried reducing $d_{model}$ to 64 and increasing heads to 8. However, this led to massive information loss. I also experimented with a complex 6-layer Encoder and 6-layer Decoder, but it resulted in Vanishing Gradients, where the model stopped learning entirely. I eventually found the "sweet spot" by simplifying the model to a 1-layer Encoder/Decoder setup to ensure stable gradient flow.
         
          5. The Padding & Class Imbalance TrapDuring early training, my dataset had excessive padding, causing the model to overfit and predict only the <pad> token. Even after applying Masking during loss calculation, the model shifted to predicting only "Spaces" because they were the most frequent simple tokens. To solve this, I implemented Token Weighting (Class Weights) during training. By assigning higher importance to actual characters over spaces and pads, I forced the model to learn the actual Hindi phonemes, which significantly improved the output quality
           
             
Currently working on implementing Beam Search to further refine the Hindi sentence structure


🏗️ Optimized Architecture: Simple yet EffectiveDue to significant Hardware and Data constraints, I strategically designed a Minimalist Transformer Architecture. My goal was not to build the "deepest" model, but the most "efficient" one that could still capture the nuances of Hindi speech within a limited computational budget.


The Core Specifications:Encoder: 1 Layer (Standard Transformer Encoder)Decoder:1 Layer (Standard Transformer Decoder)Model Dimension ($d_{model}$): 128 Attention Heads: 1 (Single-Head Attention)Feed-Forward Network (FFN): 512-dim hidden layer with ReLU activation.Why this specific configuration?After multiple experiments with deeper networks (up to 6 layers), I observed that 1-Layer Encoder and 1-Layer Decoder provided the most stable gradient flow. 

In a custom-built autograd engine, deep networks often suffer from vanishing gradients or massive memory overhead. By sticking to a single-layer setup with 128 dimensions, I achieved:Faster Convergence: The model began recognizing Hindi phonemes like "बाँधी" and "मोटा" much earlier in the training process.Resource Efficiency: Kept the Computational Graph manageable within CPU RAM limits without sacrificing the core Transformer logic (Self-Attention, Cross-Attention, and Positional Encodings).Information Integrity: Unlike my tests with 8-heads (which led to information loss at low dims), a Single-Head Attention focused the model's entire energy on the most critical audio-to-text alignments.

🔍 Technical Implementation DetailI ensured that even with a single layer, the Encoder and Decoder follow the original Transformer paper's standard:Encoder: Multi-Head Attention $\rightarrow$ Add & Norm $\rightarrow$ Feed Forward $\rightarrow$ Add & Norm.Decoder: Masked Multi-Head Attention $\rightarrow$ Add & Norm $\rightarrow$ Cross-Attention (listening to Encoder) $\rightarrow$ Add & Norm $\rightarrow$ Feed Forward $\rightarrow$ Add & Norm.
