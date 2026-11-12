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
| [Stochastic gradient descent: a noisy walk downhill](https://fleabyte26.github.io/Introduction-to-Deep-Learning/sgd.html) | The book's noisy path on x₁² + 2x₂² under constant, exponential and polynomial learning rates, plus why sampling with replacement misses about 37% of the data | 12.4 |
| [Minibatch size: noise against speed](https://fleabyte26.github.io/Introduction-to-Deep-Learning/minibatch-sgd.html) | See the scatter of minibatch gradients shrink like 1/√b, then race batch sizes 1, 10, 100 and 1,500 on the same data | 12.5 |
| [Momentum: a heavy ball that remembers](https://fleabyte26.github.io/Introduction-to-Deep-Learning/momentum.html) | The book's narrow valley 0.1x₁² + 2x₂²: plain gradient descent crawls or blows up, momentum fixes both. Includes the book's four settings | 12.6 |

Video walkthroughs on BrokenLabsFix: the [video script](VIDEO_SCRIPT.md) has Episode 1 (the first three pages) and Episode 2 (the other six).

## How to use

Click a link above to run it live, or download the `.html` file and double-click it. It works offline.

## The core idea behind all of it

1. Make a guess.
2. Score how wrong it is (the loss).
3. Nudge every weight downhill (gradient descent).
4. Repeat.

Everything in deep learning either makes this loop faster, keeps it from getting stuck, or changes the shape of the model inside it.
