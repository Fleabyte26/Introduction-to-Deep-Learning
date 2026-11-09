# Video Script: Deep Learning Without the Formula Wall

BrokenLabsFix · Kris Hornung · November 2026

## At a glance

An 11-minute screen-recorded walkthrough of the three interactive explainers in this repo. The story: one loop drives all of deep learning, and you can see it with sliders instead of formulas.

| Part | Time | Page |
| --- | --- | --- |
| Hook and intro | 0:00–1:00 | [GitHub repo](https://github.com/Fleabyte26/Introduction-to-Deep-Learning) |
| Demo 1: Symbols become numbers | 1:00–4:00 | [linear-regression.html](https://fleabyte26.github.io/Introduction-to-Deep-Learning/linear-regression.html) |
| Demo 2: Rolling downhill | 4:00–7:30 | [gradient-descent.html](https://fleabyte26.github.io/Introduction-to-Deep-Learning/gradient-descent.html) |
| Demo 3: Why sigmoid learns slowly | 7:30–10:30 | [sigmoid.html](https://fleabyte26.github.io/Introduction-to-Deep-Learning/sigmoid.html) |
| Wrap-up | 10:30–11:30 | GitHub repo |

How to read the script: each part is a table. The left column is what to do on screen, and the right column is what to say. The lines are short and plain on purpose, because YouTube auto-dubbing into Hindi and Bahasa works best on short, simple sentences with no idioms. Say them in your own words; the numbers are what matter.

Before you hit record, open these four tabs in this order: the repo, then the three pages. Reset each page so it starts at step 0.

## Opening hook and intro (0:00–1:00)

| Show | Say |
| --- | --- |
| A page of D2L formulas, or the lightboard video frame with the formulas written over the instructor | "This is how most of us are taught deep learning. Symbols. Greek letters. A formula on top of a formula." |
| Same | "I'm doing a master's in AI. I opened the textbook, and after one hour I understood nothing." |
| Switch to the gradient descent page, press the 1.05 preset, then Play. The ball flies off the chart. | "So I asked a different question. What if we just watch the numbers move?" |
| The GitHub repo page, README visible | "Today I'll show you three small pages. No install, no code. Just open the link and drag a slider." |
| Point at the three rows in the README table | "First, how a model makes a prediction. Second, how it learns. Third, why some models learn very slowly." |
| Same | "By the end, you will understand the one loop behind all of deep learning. The links are in the description." |

## Demo 1: Symbols become numbers (1:00–4:00)

Goal: the viewer sees that a formula is just a recipe, and each symbol is a slot for a number. Use the Next button instead of Play, so you control the pace.

| Show | Say |
| --- | --- |
| Open linear-regression.html, house A selected, step 1 | "This is the most basic model in machine learning: linear regression. We want to predict the price of a house." |
| Point at the formula ŷ = w₁·x₁ + w₂·x₂ + b | "This is the formula from the book. It looks scary, but it is only a recipe. Multiply each input by a weight, add them up, add a constant." |
| Click Next (step 2: 100 and 10 drop in) | "x is the house. This house is 100 square meters and 10 years old. Watch the symbols x₁ and x₂ become numbers." |
| Point at the lit rows in the table below | "The table shows what each symbol means. The rows light up when that symbol is replaced." |
| Click Next (step 3: 3, −2 and 50 drop in) | "w and b are what the model learned during training. 3 for size, minus 2 for age, and 50 as a starting price." |
| Click Next (step 4: multiply) | "Now it is just school math. 3 times 100 is 300. Minus 2 times 10 is minus 20. Older houses lose value." |
| Click Next (step 5: ŷ = 330) | "The prediction is 330. That means 330 thousand dollars." |
| Click Next (step 6: the loss formula) | "But is that a good guess? The real price was 340. We need to measure how wrong we are. That is the loss." |
| Click Next twice (steps 7 and 8: the miss is −10) | "330 minus 340 is minus 10. We missed by 10. The minus sign does not matter." |
| Click Next (step 9: square to 100) | "We square it. Squaring makes every miss positive. It also punishes big misses much more than small ones." |
| Click Next (step 10: ℓ = 50) | "Half of 100 is 50. That is the loss for this one house. The book uses half only to make the math clean later." |
| Click Next (step 11: L = 36) | "Training does this for every house and takes the average: 36. Now the model has one number that says how wrong it is." |
| Click house B, then Next a few times | "Pick another house and the same recipe gives different numbers. The formula never changes. Only the numbers change." |
| Same | "Remember the 36. The next question is: how do we make that number smaller? That is learning." |

## Demo 2: Rolling downhill (4:00–7:30)

Goal: the viewer sees that learning means stepping downhill, and that the learning rate decides whether it works. Press Reset before each preset.

| Show | Say |
| --- | --- |
| Open gradient-descent.html, preset 0.2, ball at x = 10 | "This curve is the loss. High means very wrong. The bottom, at zero, means perfect. The ball is our model, and it wants to reach the bottom." |
| Point at the short line touching the ball | "The ball cannot see the bottom. It only feels the slope under it, like walking downhill in fog." |
| Click One step (x goes 10 → 6) | "Each step follows one rule. New x equals x minus the learning rate times the slope. 10 minus 0.2 times 20 equals 6." |
| Point at the formula line with real numbers | "The same formula from the book, with real numbers inside." |
| Click One step three more times (6 → 3.6 → 2.16 → 1.3) | "Look at the steps. Big at first, then smaller and smaller. Nobody told it to slow down. The slope gets flatter near the bottom, so the steps shrink by themselves." |
| Scroll to the step table | "Every step is written out here. This is exactly what happens inside training, millions of times." |
| Preset 0.05, then Play | "Now a very small learning rate. The ball crawls. It will get there, but training takes forever." |
| Preset 0.5, then One step | "0.5 lands on the bottom in one step. Lucky. Real problems are never this neat." |
| Preset 0.9, then Play | "0.9 is too big. The ball jumps over the bottom and lands on the other side. Back and forth. It still settles, but wastes time." |
| Preset 1.05, then Play until the ball flies off | "1.05. Watch. Every jump is bigger than the last. The ball flies off the chart." |
| Same, point at the loss box | "In real training, this is when your loss suddenly becomes infinity or NaN. Now you know the first thing to check: the learning rate." |
| Point at the note under the table | "One number explains all of this: each step multiplies x by one minus two times the learning rate. You don't need to memorize it. Just remember: too small crawls, too big explodes." |

## Demo 3: Why sigmoid learns slowly (7:30–10:30)

Goal: connect Demo 2 to a real design choice. If the slope is tiny, the steps are tiny, so learning stalls. That's why modern networks use ReLU.

| Show | Say |
| --- | --- |
| Open sigmoid.html, Sigmoid tab, z = 0 | "Real networks are many layers stacked together. Between layers we put a bend, called an activation function. This S-shaped curve is the sigmoid." |
| Point at the orange tangent line and the slope box (0.250) | "In the last demo, the slope decided the step size. Here the slope is 0.25. That is the best sigmoid can ever do." |
| Drag z slowly to 2 | "Move to the right. The curve starts to flatten. The slope drops to about 0.1." |
| Drag z to 5, point at the red verdict | "At 5, the slope is 0.0066. Steps here are about 38 times smaller. The model is almost frozen." |
| Point at the shaded flat zones | "These shaded areas are where learning stalls. This problem has a name: the vanishing gradient." |
| Click the ReLU tab, drag z from 1 to 8 | "Now ReLU. Very simple: negative numbers become zero, positive numbers stay the same. On the positive side the slope is always 1. It never gets flat." |
| Drag z below 0 | "One warning: below zero the slope is zero. A neuron stuck here stops learning. People call it a dead neuron." |
| Scroll to "Now stack layers", Layers = 1, then drag to 5 | "Now the big problem. In a deep network, the slopes of all layers are multiplied together. With 5 sigmoid layers, even in the best case: 0.25 times itself 5 times. Less than 0.001." |
| Drag Layers to 10, point at the two bars | "With 10 layers, the first layer gets almost nothing. Its gradient is about one in a million. With ReLU, the bar stays full." |
| Same | "This is why deep networks were stuck for years, and why almost every modern network uses ReLU or something like it." |

## Wrap-up and call to action (10:30–11:30)

| Show | Say |
| --- | --- |
| The GitHub repo README, scroll to "The core idea behind all of it" | "So here is the whole story in three pages. One: the model makes a guess, and the loss measures how wrong it is." |
| Same | "Two: gradient descent steps downhill to make the loss smaller. The learning rate decides the step size." |
| Same | "Three: the activation function decides the slope, and the slope decides how fast each layer can learn." |
| Point at the four-step loop in the README | "Guess, measure, step downhill, repeat. That loop trains a two-number model, and it trains the biggest AI models in the world." |
| Same | "The links are in the description. Open them, drag the sliders, break things. You will learn more in ten minutes than from one hour of formulas." |
| Same | "If a topic confuses you, tell me in the comments. I'll turn it into the next page." |

## Recording checklist

- [ ] Upload the latest linear-regression.html, gradient-descent.html, sigmoid.html and README.md to the repo, and check that all three live links open
- [ ] Browser zoom at 125% so the numbers are readable on a phone
- [ ] Light mode for recording (higher contrast on video)
- [ ] Four tabs open in order: repo, linear regression, gradient descent, sigmoid
- [ ] Each page reset to its starting state
- [ ] Bookmarks bar hidden, notifications off
- [ ] Do one dry run of Demo 2 presets so the timing feels natural
- [ ] Publishing on BrokenLabsFix (Hindi/Bahasa audience, auto-dubbing on)

## Title options

1. Deep Learning Without the Formula Wall: 3 Interactive Demos
2. See How AI Learns: Gradient Descent You Can Drag
3. I Didn't Understand My AI Textbook, So I Built These 3 Demos

## Description

```
Three free interactive pages that show how deep learning works, with sliders instead of formulas. No install, no code.

1. Linear regression: symbols become numbers
https://fleabyte26.github.io/Introduction-to-Deep-Learning/linear-regression.html

2. Gradient descent: rolling downhill
https://fleabyte26.github.io/Introduction-to-Deep-Learning/gradient-descent.html

3. Why sigmoid learns slowly
https://fleabyte26.github.io/Introduction-to-Deep-Learning/sigmoid.html

All pages and source: https://github.com/Fleabyte26/Introduction-to-Deep-Learning

Based on Dive into Deep Learning (d2l.ai), sections 3.1, 3.4, 5.1, 5.3, 12.1 and 12.3.

0:00 Intro
1:00 Symbols become numbers
4:00 Gradient descent
7:30 Why sigmoid learns slowly
10:30 The one loop behind all of deep learning
```

## Tags

deep learning, machine learning, gradient descent, linear regression, sigmoid, ReLU, vanishing gradient, learning rate, neural networks, d2l, dive into deep learning, AI for beginners
