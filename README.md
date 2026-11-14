# Introduction to Deep Learning: Interactive Explainers

Small, visual explainers for the core ideas in *Dive into Deep Learning* (d2l.ai). Each one is a single HTML page you can open in a browser. Drag the controls and watch the idea work, no formulas required to get started.

## Explainers

| Page | What it shows | Book section |
|---|---|---|
| [Linear regression: symbols become numbers](https://fleabyte26.github.io/Introduction-to-Deep-Learning/linear-regression.html) | Step through the prediction and loss formulas on a real house, watching each symbol turn into a number | 3.1 |
| [Gradient descent: rolling downhill](https://fleabyte26.github.io/Introduction-to-Deep-Learning/gradient-descent.html) | Step a ball down f(x) = x² and change the learning rate to see it crawl, converge, zig-zag or explode | 3.4, 12.3 |
| [Why sigmoid learns slowly](https://fleabyte26.github.io/Introduction-to-Deep-Learning/sigmoid.html) | Drag z to see sigmoid's slope flatten toward zero (vanishing gradients), compare it with ReLU, and stack layers to see how gradients shrink | 5.1, 5.3, 12.1 |
| [Weight decay: a leash on the weights](https://fleabyte26.github.io/Introduction-to-Deep-Learning/weight-decay.html) | Train a model with 200 inputs on 20 examples, live. Raise λ and watch training loss rise, validation loss fall and the weights shrink | 3.7 |
| [Early stopping: the pattern first, the noise later](https://fleabyte26.github.io/Introduction-to-Deep-Learning/early-stopping.html) | A real network trained in your browser on points with some wrong labels. It learns the pattern first, memorizes the wrong labels later, and early stopping catches the moment in between | 5.5 |
| [Dropout: switching units off at random](https://fleabyte26.github.io/Introduction-to-Deep-Learning/dropout.html) | Watch a new random set of hidden units switch off on every minibatch, and see why dividing the survivors by 1 − p keeps the average unchanged | 5.6 |
| [Batch normalization: resetting the scale at every layer](https://fleabyte26.github.io/Introduction-to-Deep-Learning/batch-norm.html) | Send a layer values that are far off-centre or too wide and watch each minibatch get pulled back to average 0, spread 1, then rescaled by γ and β. Shows why a minibatch of 1 fails, why small minibatches are noisy, and why prediction uses running averages | 8.5 |
| [Stochastic gradient descent: a noisy walk downhill](https://fleabyte26.github.io/Introduction-to-Deep-Learning/sgd.html) | The book's noisy path on x₁² + 2x₂² under constant, exponential and polynomial learning rates, plus why sampling with replacement misses about 37% of the data | 12.4 |
| [Minibatch size: noise against speed](https://fleabyte26.github.io/Introduction-to-Deep-Learning/minibatch-sgd.html) | See the scatter of minibatch gradients shrink like 1/√b, then race batch sizes 1, 10, 100 and 1,500 on the same data | 12.5 |
| [Momentum: a heavy ball that remembers](https://fleabyte26.github.io/Introduction-to-Deep-Learning/momentum.html) | The book's narrow valley 0.1x₁² + 2x₂²: plain gradient descent crawls or blows up, momentum fixes both. Includes the book's four settings | 12.6 |
| [Four ways to close the overfitting gap](https://fleabyte26.github.io/Introduction-to-Deep-Learning/regularization-compared.html) | Four regularization techniques on one network, side by side: pick L2, dropout, early stopping or batch normalization and see where it attaches in the Keras code while the real training curves and the gap between them redraw | 3.7, 5.5, 5.6, 8.5 |
| [Padding and stride you can drag](https://fleabyte26.github.io/Introduction-to-Deep-Learning/padding-stride.html) | Drag input size, kernel size, padding and stride and watch the convolution's output size follow, with the window stepping across the grid and the formula filling in live; shows which pixels a bad stride throws away | 7.1–7.3 |
| [Where the channels go](https://fleabyte26.github.io/Introduction-to-Deep-Learning/channels.html) | Why three input channels become sixty-four output channels: set both counts and watch one output channel built from a stack of kernel slices, one per input channel, summed into a single plane; parameter count updates live | 7.4 |
| [The arrow that made networks deep](https://fleabyte26.github.io/Introduction-to-Deep-Learning/residual.html) | Why one skip connection let networks go from twenty layers to a hundred and fifty: compare a plain and a residual block, then drag depth and watch how much gradient still reaches the first block | 8.6 |
| [Image augmentation: one picture becomes many](https://fleabyte26.github.io/Introduction-to-Deep-Learning/image-augmentation.html) | Flip, crop, brighten and recolour one picture, roll the dice for a new version every epoch, and see which flips change the label. Then a real network: 92% on ankle boots, 0% when they are mirrored, until one line of augmentation fixes it | 14.1 |
| [Fine-tuning: borrow a network, replace its last layer](https://fleabyte26.github.io/Introduction-to-Deep-Learning/fine-tuning.html) | Step through the four steps of fine-tuning on ResNet-18, count how little of the model starts from nothing, and see why the copied layers and the new output layer need different learning rates | 14.2 |

Video walkthroughs on BrokenLabsFix: [Episode 1, covering the first three pages](https://www.youtube.com/watch?v=YQF4lMUv9d4), [Episode 2, covering the six newer pages](https://www.youtube.com/watch?v=_69pXSvmCOY), [Episode 3, on batch normalization](https://www.youtube.com/watch?v=r_NqjxmXD8g) [Episode 4, comparing the four regularization techniques](https://www.youtube.com/watch?v=b4tLpx3_934) and [Episode 5, on convolutions, channels and residual connections](https://www.youtube.com/watch?v=8Ab74b_z-Es). The [video script](VIDEO_SCRIPT.md) has a script for each episode, including Episode 6 on image augmentation and fine-tuning.

## How to use

Click a link above to run it live, or download the `.html` file and double-click it. It works offline.

## The core idea behind all of it

1. Make a guess.
2. Score how wrong it is (the loss).
3. Nudge every weight downhill (gradient descent).
4. Repeat.

Everything in deep learning either makes this loop faster, keeps it from getting stuck, or changes the shape of the model inside it.
