# Introduction to Deep Learning: Interactive Explainers

Small, visual explainers for the core ideas in *Dive into Deep Learning* (d2l.ai). Each one is a single HTML page you can open in a browser. Drag the controls and watch the idea work, no formulas required to get started.

## Explainers

| Page | What it shows | Book section |
|---|---|---|
| [Linear regression: symbols become numbers](https://fleabyte26.github.io/Introduction-to-Deep-Learning/linear-regression.html) | Step through the prediction and loss formulas on a real house, watching each symbol turn into a number | 3.1 |
| [Gradient descent: rolling downhill](https://fleabyte26.github.io/Introduction-to-Deep-Learning/gradient-descent.html) | Step a ball down f(x) = x² and change the learning rate to see it crawl, converge, zig-zag or explode | 3.4, 12.3 |
| [Why sigmoid learns slowly](https://fleabyte26.github.io/Introduction-to-Deep-Learning/sigmoid.html) | Drag z to see sigmoid's slope flatten toward zero (vanishing gradients), compare it with ReLU, and stack layers to see how gradients shrink | 5.1, 5.3, 12.1 |

Video walkthrough of all three on BrokenLabsFix: see the [video script](VIDEO_SCRIPT.md).

## How to use

Click a link above to run it live, or download the `.html` file and double-click it. It works offline.

## The core idea behind all of it

1. Make a guess.
2. Score how wrong it is (the loss).
3. Nudge every weight downhill (gradient descent).
4. Repeat.

Everything in deep learning either makes this loop faster, keeps it from getting stuck, or changes the shape of the model inside it.
