# Video Script: Deep Learning Without the Formula Wall

BrokenLabsFix · Kris Hornung · November 2026

Watch the recorded video: https://www.youtube.com/watch?v=YQF4lMUv9d4

This file holds four scripts. Episode 1, below, covers the first three pages. [Episode 2](#episode-2-how-models-stop-memorizing-and-learn-faster) covers the six newer pages: weight decay, early stopping, dropout, SGD, minibatch size and momentum. [Episode 3](#episode-3-batch-normalization-you-can-drag) covers the batch normalization page. [Episode 4](#episode-4-four-ways-to-close-the-overfitting-gap) covers the regularization comparison page.

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

---

# Episode 2: How Models Stop Memorizing and Learn Faster

BrokenLabsFix · Kris Hornung · November 2026

Watch the recorded video: https://www.youtube.com/watch?v=_69pXSvmCOY

## Episode 2 at a glance

An 18-minute screen-recorded walkthrough of the six newer explainers. Episode 1 ended with one loop: guess, measure, step downhill, repeat. This episode asks two questions about that loop. First: how do we stop the model from memorizing? Second: how do we make each step cheaper and faster?

| Part | Time | Page |
| --- | --- | --- |
| Hook and intro | 0:00–1:00 | [early-stopping.html](https://fleabyte26.github.io/Introduction-to-Deep-Learning/early-stopping.html), then the [GitHub repo](https://github.com/Fleabyte26/Introduction-to-Deep-Learning) |
| Demo 1: Weight decay | 1:00–3:45 | [weight-decay.html](https://fleabyte26.github.io/Introduction-to-Deep-Learning/weight-decay.html) |
| Demo 2: Early stopping | 3:45–6:30 | [early-stopping.html](https://fleabyte26.github.io/Introduction-to-Deep-Learning/early-stopping.html) |
| Demo 3: Dropout | 6:30–9:00 | [dropout.html](https://fleabyte26.github.io/Introduction-to-Deep-Learning/dropout.html) |
| Demo 4: Stochastic gradient descent | 9:00–12:00 | [sgd.html](https://fleabyte26.github.io/Introduction-to-Deep-Learning/sgd.html) |
| Demo 5: Minibatch size | 12:00–14:45 | [minibatch-sgd.html](https://fleabyte26.github.io/Introduction-to-Deep-Learning/minibatch-sgd.html) |
| Demo 6: Momentum | 14:45–17:30 | [momentum.html](https://fleabyte26.github.io/Introduction-to-Deep-Learning/momentum.html) |
| Wrap-up | 17:30–18:30 | GitHub repo |

The format is the same as Episode 1: the left column is what to do on screen, and the right column is what to say, in short plain sentences for auto-dubbing.

About the numbers: every page starts from a fixed random seed, so a freshly loaded page gives the numbers in this script. Do not press New data or New random run before the lines that quote a number, because those buttons change the numbers. If a number on screen differs from the script, read the one on screen.

If 18 minutes is too long for one video, it splits cleanly after Demo 3. Demos 1 to 3 are about memorizing. Demos 4 to 6 are about faster training.

## Episode 2 hook and intro (0:00–1:00)

Before recording, open early-stopping.html in a spare tab, press Play and let it run to step 1500. The hook starts on that finished screen.

| Show | Say |
| --- | --- |
| early-stopping.html, already run to step 1500. Point at the shading wrapped around the ringed points | "This network has a problem. Some of its training data is wrong. And it learned the wrong answers perfectly." |
| Point at "Wrong labels memorized 88%" and "Accuracy on new points 81.0%" | "It memorized 88 percent of the mistakes. On new data, its accuracy fell from almost 95 percent to 81." |
| The GitHub repo README, scroll to the table | "In the last video we saw the one loop behind deep learning. Guess, measure, step downhill, repeat." |
| Point at the six new rows | "Today, six more pages, and two questions. One: how do we stop a model from memorizing? Two: how do we make training faster?" |
| Same | "No install, no code. The links are in the description. Open them and follow along." |

## Demo 1: Weight decay (1:00–3:45)

Goal: the viewer sees overfitting as a gap between two curves, and sees weight decay close part of that gap by keeping the weights small.

| Show | Say |
| --- | --- |
| Open weight-decay.html, λ = 0, epoch 0 | "This model has 200 inputs, but only 20 training examples. That is like a student with 20 practice questions and 200 pages to write notes on. It can simply memorize." |
| Point at the two legends under the left chart | "Two curves. Blue is the training loss: the questions the model has seen. Orange is the validation loss: 100 questions it has never seen." |
| Click One epoch ten times, slowly | "No weight decay. Each click is one pass through the 20 examples. Watch the blue curve. It dives." |
| Point at the Training loss and Validation loss boxes | "After 10 epochs, training loss is almost zero. But validation loss has not moved. It stays near 0.019." |
| Scroll to the table, point at the Gap column on the epoch 10 row | "The model is 1,700 times worse on new data than on its own training data. This gap has a name: overfitting." |
| Press Play and let it run to epoch 30 | "Training longer does not help. The blue curve keeps diving, and the orange curve stays flat." |
| Point at the right chart, the bars and the green line | "These bars are the 200 weights. The green line is the true value. The weights are scattered everywhere, because 20 examples cannot pin down 200 weights." |
| Click the preset "3 the book's value" | "Now weight decay. The idea is simple. Before every update, shrink every weight a little." |
| Point at the formula line: 1 − 0.01 × 3.0 = 0.970 | "With lambda equal to 3, each update keeps 97 percent of every weight. Then the data pushes it. A weight stays large only if the data keeps asking for it." |
| Click One epoch ten times | "Ten epochs again. The blue curve does not dive this time. It levels off. The model is not allowed to memorize." |
| Point at the Penalty box (0.00168), then the note under the table | "The book reports the size of the weights after 10 epochs: 0.0017. This page gives 0.00168. It is the same experiment, running live in your browser." |
| Press Play and let it run to epoch 30. Point at the Validation loss box (0.0111) and the bars | "And the validation loss drops from 0.019 to 0.011. Look at the bars: the weights are much smaller now." |
| Click the preset "10 heavy", press Play | "Can we use more? With lambda equal to 10, the weights are squeezed almost to zero. Training loss stays high, and validation is no better than before." |
| Point at the verdict box | "Too much decay and the model cannot learn the real pattern either. That is called underfitting. Lambda is a dial you must tune." |

## Demo 2: Early stopping (3:45–6:30)

Goal: the viewer sees that a network learns the real pattern first and the mistakes later, so stopping at the right moment is a form of protection. Press Reset before you start, so the page is back at step 0.

| Show | Say |
| --- | --- |
| Open early-stopping.html, Reset, Wrong labels at 20% | "Here is a different way to fight memorizing. Two classes of points, blue and orange, in two curved shapes." |
| Point at the ringed points | "The points with a ring have the wrong label, on purpose. 20 percent of them. Real data always has mistakes like this." |
| Point at the note at the bottom: 200 hidden units, 80 points | "The network has 200 hidden units and only 80 training points. It is big enough to memorize every one of them." |
| Press Play. Pause after one or two seconds | "Watch the shading. That is what the network predicts. First it finds the two big shapes. The real pattern comes first." |
| Point at "Correct labels fitted" and "Wrong labels memorized" | "It already fits almost all the correct points, and only a few of the wrong ones." |
| Press Play again and let it run to step 1500 | "Now keep training. The shading starts to bend. Small islands appear around the ringed points. The network is now learning the mistakes." |
| Point at the right chart: blue falling, orange rising | "Look at the loss. Training loss keeps falling. Validation loss turns around and climbs. That turn is the moment memorizing begins." |
| Point at the green dot and the dashed red line | "Early stopping watches the validation loss. The green dot is the best moment: step 60. When there is no improvement for a while, it stops, and it keeps the model from step 60." |
| Scroll to the comparison table | "Compare the two models. Stopped early: 94.7 percent on new points, and 6 percent of the wrong labels memorized." |
| Same, point at the second row | "Trained to the end: 81 percent on new points, and 88 percent of the wrong labels memorized. More training made the model worse." |
| Point at the Patience slider | "Patience is how long we wait. The page checks every 10 steps. Patience 8 means: 8 checks with no improvement, then stop." |
| Drag Wrong labels to 0%, press Play, let it finish | "One more test. No wrong labels at all. Now the two models are almost the same: 99.7 and 99.3 percent. With clean data, early stopping saves time but changes little." |
| Point at the notes | "A note for students: the book explains this idea in section 5.5 but gives no experiment. This demo is my own." |

## Demo 3: Dropout (6:30–9:00)

Goal: the viewer sees what dropout does on each minibatch, and understands why the survivors are multiplied. Use One minibatch at first, so you control the pace.

| Show | Say |
| --- | --- |
| Open dropout.html, p = 0.50, nothing drawn yet | "The third tool against memorizing is the strangest one. During training, we switch off parts of the network at random." |
| Click One minibatch | "One minibatch. The crossed-out units are off. They send nothing forward." |
| Click One minibatch two more times | "Next minibatch, a different set. And again, a different set. Every minibatch trains a slightly different network." |
| Point at the network | "Why does this help? A unit cannot depend on one special partner, because that partner may be off next time. Each unit must be useful by itself." |
| Point at the formula line and "Survivors multiplied by 2.00" | "There is one detail. With p equal to 0.5, half of the units are off. So the units that stay on are multiplied by 2." |
| Scroll to the bar chart. Point at one unit that is off and one that is on | "Grey is the original value. Orange is the value after dropout. It is either zero, or double." |
| Click Draw 1,000 at once. Point at the blue bars | "Now a thousand minibatches. Blue is the average. Half the time zero, half the time double. The average comes back to the original value." |
| Point at "Average is off by" (about 2 to 3 percent) | "The average is only two or three percent away. So dropout adds noise, but it does not change the average signal. That is why we divide by one minus p." |
| Click "Prediction: dropout off" | "When training is finished, dropout is switched off. Every unit is on, and nothing is multiplied. Dropout is only for training." |
| Click "Training: dropout on", drag p to 0.80, click One minibatch | "What if we drop 80 percent? Most of the layer is gone, and the survivors are multiplied by 5. Very noisy. Training becomes slow and unstable." |
| Drag p back to 0.50 | "The book uses 0.5 for hidden layers, and a smaller value close to the input." |

## Demo 4: Stochastic gradient descent (9:00–12:00)

Goal: turn from memorizing to speed. The viewer sees that a cheap, noisy step still goes downhill, and that the learning rate must shrink over time, but not too fast.

| Show | Say |
| --- | --- |
| Open sgd.html, Constant selected, step 0 | "Now the second question: speed. In the last video, the ball rolled down a smooth curve. But to get that smooth slope, the model must look at every training example before each step." |
| Same | "With millions of examples, that is very slow. So we cheat. We look at one example, and take a step right away. That is stochastic gradient descent." |
| Point at the rings and the green dot | "This is a bowl, seen from above. The green dot is the bottom. We start far away, at minus 5, minus 2." |
| Click One step three times. Point at the formula line and the word "noise" | "One example gives only a rough slope. Here it is in numbers: the true slope, plus some noise." |
| Press Play and let it run to step 100 | "The dashed grey line is the path with no noise. The orange path is the real one. It shakes, but it still goes downhill." |
| Point at the cloud of dots near the goal and the verdict | "But look at the end. It reached the bottom of the bowl, and now it cannot settle. It is still 0.55 away, and it keeps wandering. Every step is still the same size." |
| Click Exponential decay, press Play | "The fix is to make the learning rate smaller over time. First try: shrink it very fast." |
| Point at the "Learning rate at each step" chart, then the verdict | "The shaking is gone. But the ball stopped too early. The learning rate is almost zero, and we are stuck 0.89 from the goal." |
| Click Polynomial decay, press Play | "Second try: shrink it slowly." |
| Point at the "Distance from the goal" chart and the verdict | "Now it keeps moving and it calms down. 0.23 from the goal, and still getting closer. This is why real training uses a learning rate schedule." |
| Scroll to "Does each pass see every example?" Click "Pick 100 at random (with replacement)" and wait for it to finish | "One more surprise. There are 100 training examples here. Pick 100 times at random." |
| Point at "Different examples seen" and "Never seen" | "We saw only 61 of them. 39 were never picked, and others were picked two or three times. The math says about 63 percent, always." |
| Click Reset in this panel, then "One shuffled pass (without replacement)" | "So real training code does this: shuffle the data, then walk through it once. Every example is seen exactly one time. That is one epoch." |

## Demo 5: Minibatch size (12:00–14:45)

Goal: the viewer sees the trade between one example (noisy) and all examples (slow), and why a batch in the middle wins.

| Show | Say |
| --- | --- |
| Open minibatch-sgd.html, slider at b = 1 | "One example per step is noisy. All examples per step is slow. The answer is in the middle: a minibatch." |
| Point at the cloud of orange dots and the green dot | "Each orange dot is the slope that one minibatch reports. The green dot in the middle is the true slope from all 1,500 examples. With batch size 1, the dots are everywhere." |
| Drag the slider to b = 10 | "Batch size 10. We average 10 examples. The cloud shrinks to about one third." |
| Drag the slider to b = 100. Point at "Noise relative to b = 1" | "Batch size 100. The noise is about 10 percent of what it was. The rule: noise falls with the square root of the batch size." |
| Drag the slider to b = 1,500 | "All 1,500. No noise at all. But every step costs 1,500 examples." |
| Drag back to 100, point at the verdict | "Look at the price. From 1 to 100 examples, the noise falls ten times. From 100 to 1,500, we pay fifteen times more, and there is very little noise left to remove." |
| Scroll to "The race". Press Play, then pause after about one second | "Now a race. Four batch sizes, the same data. Look at the full batch, in blue. It has not moved. It must read all 1,500 examples before its first step." |
| Point at the "Updates so far" column | "Batch size 1 has already made hundreds of small steps." |
| Press Play and let it finish (6 passes) | "After six passes through the data, the three small batches have all arrived, at a loss near 0.118. The full batch has made only six steps, and it is still at 0.250." |
| Point at the last column: "Book: seconds per epoch" | "So is batch size 1 the winner? No. This column is the real time the book measured. Batch size 1 needs 0.685 seconds for each pass. Batch size 100 needs 0.025." |
| Same | "The computer is very fast when it processes many examples together, and slow when it takes them one at a time. Batch size 1 is 34 times slower for each pass." |
| Same | "So the middle wins. Enough examples to calm the noise and use the hardware. Not so many that you wait for every step." |
| Point at the last note | "A note: the book uses a real dataset with more inputs. This page uses a small dataset with two weights, so we can draw the path. The batch sizes, learning rates and timings are the book's." |

## Demo 6: Momentum (14:45–17:30)

Goal: the viewer sees a problem that no single learning rate can solve, and sees momentum solve it. Use the four preset buttons in order. Each run stops by itself after about 40 steps.

| Show | Say |
| --- | --- |
| Open momentum.html. Click the first preset: "η 0.4, no momentum" | "The last tool. This valley is long and narrow. Left to right, the floor is almost flat. Up and down, the walls are steep." |
| Press Play and let it finish | "Plain gradient descent, learning rate 0.4. Up and down, it reaches the floor quickly. But along the floor it crawls." |
| Point at the x₁ box and the verdict | "After 40 steps, x1 is still not at the goal. Each step removes only 8 percent of the distance that is left." |
| Click the second preset: "η 0.6, no momentum". Point at the red warning | "So take bigger steps? Learning rate 0.6. The page gives a warning before we even start." |
| Press Play and let it finish | "The ball bounces off the steep walls, and every bounce is bigger. Within 20 steps, x2 is past minus 1,700, and it keeps growing. This is the explosion from the last video." |
| Same | "So we are stuck. A small step crawls along the floor. A big step explodes on the walls. No single learning rate fixes both." |
| Click the third preset: "η 0.6, β 0.5". Press Play and let it finish | "Now momentum. The same learning rate, 0.6, the one that exploded. But now the ball remembers its past steps." |
| Point at the orange path against the dashed grey path | "It works. The ball goes straight to the goal. The dashed grey line is the same learning rate without momentum." |
| Press Reset, then click One step 20 times. Point at the x₁ and x₂ boxes (0.007 and 0.003), then the note under the charts | "The book gives its result after 20 steps: x1 is 0.007188 and x2 is 0.002553. This page gives the same numbers." |
| Scroll to "What remembering means". Point at the bars and "Gradients remembered 2.0" | "How does it work? Each step uses an average of the recent slopes. With beta 0.5, the last slope counts fully, the one before counts half, then a quarter. That is like remembering 2 slopes." |
| Point back at the path | "On the steep walls, the slopes point up, then down, then up. In the average, they cancel. Along the floor, they all point the same way, so they add up. Less bouncing, more travel." |
| Click the fourth preset: "η 0.6, β 0.25". Press Play | "With less momentum, beta 0.25, it still arrives, but only just. It keeps swinging for a long time." |
| Drag the β slider to 0.90, point at "Gradients remembered 10.0" | "And beta 0.9 remembers about 10 slopes. That is the usual value in real training." |

## Episode 2 wrap-up and call to action (17:30–18:30)

| Show | Say |
| --- | --- |
| The GitHub repo README, the explainer table | "Six pages, two questions. First question: how do we stop a model from memorizing?" |
| Point at the weight decay, early stopping and dropout rows | "Weight decay keeps the weights small. Early stopping ends training before the model learns the mistakes. Dropout makes sure no unit depends on another." |
| Point at the SGD, minibatch and momentum rows | "Second question: how do we train faster? Take cheap noisy steps and shrink the learning rate slowly. Use a minibatch, not one example and not all of them. And add momentum, so the bounces cancel and the progress adds up." |
| Scroll to "The core idea behind all of it" | "The loop is still the same. Guess, measure, step downhill, repeat. Everything today makes that loop safer or faster." |
| Same | "The links are in the description. Open the pages and change the numbers. Try to break them. That is how I learned this." |
| Same | "Tell me in the comments which topic confuses you. I'll turn it into the next page." |

## Episode 2 recording checklist

- [ ] Check that all six live links open (they went live when the six pages were merged on November 12)
- [ ] Browser zoom at 125% so the numbers are readable on a phone
- [ ] Light mode for recording (higher contrast on video)
- [ ] Tabs open in order: early stopping (run to step 1500 for the hook), repo, weight decay, early stopping (fresh), dropout, SGD, minibatch, momentum
- [ ] Reload every page except the hook tab, so each one starts from its fixed seed and matches the numbers in this script
- [ ] Do not press New data or New random run during the takes that quote numbers
- [ ] Bookmarks bar hidden, notifications off
- [ ] Do one dry run of the minibatch race and the early stopping run, so you know how long each takes to finish
- [ ] Publishing on BrokenLabsFix (Hindi/Bahasa audience, auto-dubbing on)

## Episode 2 title options

1. Why AI Memorizes Instead of Learning: 6 Interactive Demos
2. Deep Learning Without the Formula Wall, Part 2: Stop Memorizing, Train Faster
3. Overfitting, Dropout, SGD and Momentum, With Sliders Instead of Formulas

## Episode 2 description

```
Six free interactive pages that show how a neural network stops memorizing and trains faster, with sliders instead of formulas. No install, no code.

1. Weight decay: a leash on the weights
https://fleabyte26.github.io/Introduction-to-Deep-Learning/weight-decay.html

2. Early stopping: the pattern first, the noise later
https://fleabyte26.github.io/Introduction-to-Deep-Learning/early-stopping.html

3. Dropout: switching units off at random
https://fleabyte26.github.io/Introduction-to-Deep-Learning/dropout.html

4. Stochastic gradient descent: a noisy walk downhill
https://fleabyte26.github.io/Introduction-to-Deep-Learning/sgd.html

5. Minibatch size: noise against speed
https://fleabyte26.github.io/Introduction-to-Deep-Learning/minibatch-sgd.html

6. Momentum: a heavy ball that remembers
https://fleabyte26.github.io/Introduction-to-Deep-Learning/momentum.html

All pages and source: https://github.com/Fleabyte26/Introduction-to-Deep-Learning

Based on Dive into Deep Learning (d2l.ai), sections 3.7, 5.5, 5.6, 12.4, 12.5 and 12.6.

0:00 Intro
1:00 Weight decay
3:45 Early stopping
6:30 Dropout
9:00 Stochastic gradient descent
12:00 Minibatch size
14:45 Momentum
17:30 Two questions, six answers
```

## Episode 2 tags

deep learning, machine learning, overfitting, weight decay, L2 regularization, early stopping, dropout, stochastic gradient descent, SGD, minibatch, batch size, momentum, learning rate schedule, neural networks, d2l, dive into deep learning, AI for beginners

# Episode 3: Batch Normalization You Can Drag

BrokenLabsFix · Kris Hornung · November 2026

Watch the recorded video: https://www.youtube.com/watch?v=r_NqjxmXD8g

Earlier episodes: [Episode 1](https://www.youtube.com/watch?v=YQF4lMUv9d4) · [Episode 2](https://www.youtube.com/watch?v=_69pXSvmCOY)

## Episode 3 at a glance

A 9-minute screen-recorded walkthrough of one page: batch normalization, from Section 8.5 of *Dive into Deep Learning*. The story: deep inside a network the numbers drift to bad places, and one small layer puts them back, every minibatch.

| Part | Time | What happens |
| --- | --- | --- |
| Hook and intro | 0:00–0:45 | The numbers arrive in the wrong place |
| Part 1: The problem | 0:45–2:00 | Values in the sigmoid's flat zone |
| Part 2: Subtract and divide | 2:00–3:30 | Any input becomes average 0, spread 1 |
| Part 3: γ and β | 3:30–4:45 | The network picks its own scale and shift |
| Part 4: The same example, different company | 4:45–7:00 | Minibatch size, noise, and why size 1 fails |
| Part 5: Training and prediction | 7:00–8:30 | Running averages |
| Wrap-up | 8:30–9:15 | The formula, read aloud in plain words |

Page: [batch-norm.html](https://fleabyte26.github.io/Introduction-to-Deep-Learning/batch-norm.html)

How to read the script: same as before. The left column is what to do on screen, the right column is what to say. The lines are short and plain on purpose, for YouTube auto-dubbing into Hindi and Bahasa. Say them in your own words; the numbers are what matter.

About the numbers: the page uses a fixed random seed, so every number below appears exactly as written, as long as you follow the clicks in order. Moving the average, spread or minibatch size slider restarts the page from zero. Moving γ or β does not. Use "Draw 500 at once" where the script says so, not Play, because Play gives different numbers depending on when you pause.

## Episode 3 hook and intro (0:00–0:45)

| Show | Say |
| --- | --- |
| The batch norm page as it loads: average 5, spread 2.5, minibatch 32. Top row of dots sits far right, in the shaded zone | "Look at the top row. Each dot is one number arriving at a layer, deep inside a network. Most of them are in the shaded zone. That is a bad place to be." |
| Point at the bottom row, centred on zero | "Now look at the bottom row. The same numbers, after one small layer. All of them are back in the middle." |
| Same | "That layer is called batch normalization. Almost every deep network for images uses it. Today we open it and see how it works." |
| Same | "This is Section 8.5 of the book Dive into Deep Learning. One page, five sliders, no code. The link is in the description." |

## Part 1: The problem (0:45–2:00)

Goal: the viewer sees why badly placed numbers stop learning. This connects to the sigmoid page from Episode 1.

| Show | Say |
| --- | --- |
| Point at the top row, then at the first two sliders (5.0 and 2.5) | "The top row is what arrives. These values have an average of 5 and a spread of 2.5. Nobody chose that. It is just what the earlier layers produced." |
| Point at the shaded areas beyond −4 and +4 | "The shaded areas are beyond plus 4 and minus 4. In Episode 1 we saw the sigmoid curve. Out here the curve is flat. The slope is almost zero." |
| Point at the stat box "In sigmoid's flat zone: before, after" showing 66% → 0% | "A flat slope means a tiny step. A tiny step means no learning. Here, 66 percent of the values are in the flat zone." |
| Same | "And it gets worse. The earlier layers are still training. So these numbers keep moving. The layer is aiming at a target that never stands still." |
| Point at the ringed dot in the top row, at 7.5 | "One dot has a ring. That is one example, and we will follow it for the whole video. It arrives at 7.5." |

## Part 2: Subtract and divide (2:00–3:30)

Goal: the viewer sees that the middle row is always average 0, spread 1, whatever arrives.

| Show | Say |
| --- | --- |
| Point at the formula line: y = γ × (x − 4.93) / 2.53 + β | "Here is the formula, with real numbers inside. Two highlighted numbers: 4.93 and 2.53." |
| Point at the stat box "This minibatch: average, spread" showing 4.93, 2.53 | "They are the average and the spread of these 32 values. The layer measures them itself, from this minibatch." |
| Point at the middle row (blue dots) | "Step one: subtract the average. Now the values are centred on zero. Step two: divide by the spread. Now they have a spread of 1." |
| Point at the ringed dot, top row then middle row | "Our ringed example was 7.5. Minus 4.93, divided by 2.53. It lands at about 1." |
| Drag Incoming average slowly from 5 to −6 | "Now I make the input worse. I move the average to minus 6. The top row slides to the left. 81 percent in the flat zone." |
| Point at the middle and bottom rows, which did not move | "The other two rows did not move. The layer subtracted the new average, and the result is the same." |
| Drag Incoming spread down to 0.2 | "Now a very narrow spread, 0.2. All the values are squeezed together. 100 percent in the flat zone. After the layer, zero percent." |
| Drag the average back to 5 and the spread back to 2.5 | "This is the first idea. Whatever arrives, after subtract and divide, the average is 0 and the spread is 1." |

## Part 3: γ and β (3:30–4:45)

Goal: the viewer understands that the layer does not force 0 and 1 forever. It hands the choice to two learned numbers.

| Show | Say |
| --- | --- |
| Point at the bottom row (orange dots) and the two sliders γ and β | "But maybe average 0 and spread 1 is not what the network needs. So there is a third row, and two more numbers. Gamma and beta." |
| Drag Scale γ from 1 to 2 | "Gamma is the scale. I set it to 2, and the bottom row gets wider. The output spread is now exactly 2." |
| Drag Shift β from 0 to 3, then γ to 3 | "Beta is the shift. I set it to 3, and the row moves to the right. Now gamma 3, beta 3. The output has average 3 and spread 3." |
| Point at the yellow verdict box: 34% of the output in the flat zone | "Look at the message. 34 percent of the output is in the flat zone again. Nothing forbids that." |
| Same | "So what did we gain? Before, the scale was decided by every layer upstream, all changing at once. Now it is decided by two numbers, right here." |
| Same | "And in a real network, nobody drags these sliders. Gamma and beta are learned by gradient descent, like any other weight." |
| Drag γ back to 1.0 and β back to 0.0 | "I put them back to 1 and 0." |

## Part 4: The same example, different company (4:45–7:00)

Goal: the viewer sees that an example's output depends on the other examples in its minibatch, and that minibatch size controls how much. Scroll so the strips and the lower chart are both visible if you can.

| Show | Say |
| --- | --- |
| Click One minibatch. The dots change, the ringed dot stays at 7.5 on top | "Now a new minibatch. 31 new examples, plus our ringed one. Its own value did not change. Still 7.5." |
| Point at the formula line: now 5.11 and 1.92 | "But the average and the spread changed, because the other examples changed. Now 5.11 and 1.92." |
| Click One minibatch two more times. Point at the ringed dot in the bottom row each time | "So its output changes too. 1.25. Then 0.52. Then 0.89. Same example, three different answers." |
| Scroll to the chart "The same example, different company" | "This chart follows only the ringed example. Each point is its output on one minibatch. The dashed line is where it would sit with perfect statistics." |
| Click Draw 500 at once | "Here are 500 minibatches. The points wobble around the line. The wobble is about plus or minus 0.21." |
| Same | "This wobble is noise. Remember dropout from Episode 2? Random noise during training made the model more robust. This does a little of the same, for free." |
| Drag Minibatch size to 128, click Draw 500 at once | "A bigger minibatch, 128. The statistics are more stable. The wobble drops to 0.11." |
| Drag Minibatch size to 4, click Draw 500 at once | "Now only 4 examples. The wobble is 0.51. The average of 4 numbers is a rough guess, and it jumps around." |
| Drag Minibatch size to 2, click Draw 500 at once | "With 2 examples, something strange. The output is only ever plus 1 or minus 1. With two values, one is always above the average and one is below." |
| Drag Minibatch size to 1. Point at the red verdict box | "And a minibatch of 1. The average of one number is the number itself. Subtract it, and you get zero. Every time." |
| Point at the bottom row: a single dot at 0 | "The output is just beta, whatever arrived. The layer threw away all the information. The book gives this warning: batch normalization needs more than one example per minibatch." |

## Part 5: Training and prediction (7:00–8:30)

Goal: the viewer understands why the layer has two modes, and what the running averages are for.

| Show | Say |
| --- | --- |
| Drag Minibatch size back to 32 | "This creates a problem. When the model is finished, a user may send one single image. There is no minibatch to measure." |
| Point at the stat box "Running averages kept for prediction", showing 0.00, 1.00 | "The answer is in this box. During training, the layer keeps a running average of the averages and the spreads it sees. It starts at 0 and 1." |
| Click "Prediction: use the running averages". Point at the yellow verdict and 66% → 66% | "I switch to prediction mode now, before any training. The layer subtracts 0 and divides by 1. So it does nothing. 66 percent in, 66 percent out." |
| Click Draw 500 at once (the page goes back to training mode by itself) | "So we train. 500 minibatches. Each one moves the running averages one tenth of the way." |
| Point at the running averages box: 5.15, 2.42 | "Now they are 5.15 and 2.42. Close to the true values, 5 and 2.5." |
| Click "Prediction: use the running averages". Point at the formula line: 5.15 and 2.42 | "Prediction mode again. The formula now uses the saved numbers. They are fixed. They do not change with the minibatch." |
| Point at the stat box "Ringed example's output wobble", showing 0 | "So the wobble is zero. One example gives the same answer every time, whatever else is in the batch. That is what you want from a finished model." |
| Same | "This is the practical lesson. A batch normalization layer acts differently in training and in prediction, the same as dropout. If you forget to switch your model to prediction mode, your results will be wrong." |

## Episode 3 wrap-up and call to action (8:30–9:15)

| Show | Say |
| --- | --- |
| Scroll to "What to take from this". Point at the formula BN(x) = γ × (x − μ) / σ + β | "Here is the formula from the book. Now you can read it. Take x. Subtract the minibatch average. Divide by the minibatch spread. Then scale by gamma and shift by beta." |
| Same | "Three things to remember. One: whatever arrives, the layer resets it to average 0 and spread 1, then lets the network choose the scale." |
| Same | "Two: in training it uses the current minibatch, so small minibatches are noisy, and a minibatch of 1 does not work." |
| Same | "Three: in prediction it uses running averages saved during training." |
| Point at the last note on the page | "With this layer, deep networks train faster and accept bigger learning rates. Why it works so well is still argued about. The book says so plainly, and I like that honesty." |
| The GitHub repo README | "The link is in the description. Open it, drag the sliders, try a minibatch of 1. Tell me in the comments which topic you want next." |

## Episode 3 recording checklist

- [ ] Check that the live link opens: https://fleabyte26.github.io/Introduction-to-Deep-Learning/batch-norm.html
- [ ] Reload the page before recording so it starts at average 5, spread 2.5, minibatch 32, γ 1, β 0, training mode
- [ ] Browser zoom at 125% so the numbers are readable on a phone
- [ ] Light mode for recording (higher contrast on video)
- [ ] Bookmarks bar hidden, notifications off
- [ ] Do one dry run of Part 4. It has the most clicks, and each minibatch size change restarts the page
- [ ] In Parts 4 and 5 use "Draw 500 at once", not Play, so the numbers match the script
- [ ] Publishing on BrokenLabsFix (Hindi/Bahasa audience, auto-dubbing on)

## Episode 3 title options

1. Batch Normalization You Can Drag: No Formulas
2. Batch Normalization Explained With Sliders (Dive into Deep Learning 8.5)
3. Why Deep Networks Need Batch Normalization: See It Happen

## Episode 3 description

```
One free interactive page that shows how batch normalization works, with sliders instead of formulas. No install, no code.

Batch normalization: resetting the scale at every layer
https://fleabyte26.github.io/Introduction-to-Deep-Learning/batch-norm.html

All pages and source: https://github.com/Fleabyte26/Introduction-to-Deep-Learning

Episode 1 (linear regression, gradient descent, sigmoid): https://www.youtube.com/watch?v=YQF4lMUv9d4
Episode 2 (weight decay, early stopping, dropout, SGD, minibatch size, momentum): https://www.youtube.com/watch?v=_69pXSvmCOY

Based on Dive into Deep Learning (d2l.ai), Chapter 8: Modern CNNs, Section 8.5: Batch Normalization.

0:00 Numbers in the wrong place
0:45 The problem: the flat zone
2:00 Subtract the average, divide by the spread
3:30 Gamma and beta
4:45 Minibatch size and noise
7:00 Training mode and prediction mode
8:30 The formula in plain words
```

## Episode 3 tags

batch normalization, batch norm, deep learning, dive into deep learning, d2l, modern CNNs, vanishing gradient, minibatch, running average, training vs inference, model.eval, neural networks, machine learning tutorial, interactive explainer

# Episode 4: Four Ways to Close the Overfitting Gap

BrokenLabsFix · Kris Hornung · November 2026

Earlier episodes: [Episode 1](https://www.youtube.com/watch?v=YQF4lMUv9d4) · [Episode 2](https://www.youtube.com/watch?v=_69pXSvmCOY) · [Episode 3](https://www.youtube.com/watch?v=r_NqjxmXD8g)

## Episode 4 at a glance

A 9-minute walkthrough of one page that puts four regularization techniques side by side on the same network. The story: they are not four unrelated tricks, they are four places to plug something into the same five lines of Keras.

| Part | Time | What happens |
| --- | --- | --- |
| Hook: the gap | 0:00–0:45 | Two curves that drift apart |
| Part 1: The skeleton | 0:45–2:00 | Build, compile, fit |
| Part 2: The baseline | 2:00–3:15 | Press Play, watch 7.7 points open up |
| Part 3: L2 | 3:15–4:15 | An argument inside the layer |
| Part 4: Dropout | 4:15–5:30 | A layer between the layers |
| Part 5: Early stopping | 5:30–6:45 | A callback on fit |
| Part 6: Batch normalization | 6:45–8:00 | The one that goes the wrong way |
| Wrap-up | 8:00–9:00 | What the bars say |

Page: [regularization-compared.html](https://fleabyte26.github.io/Introduction-to-Deep-Learning/regularization-compared.html)

Same format as before: left column is what to do on screen, right column is what to say. Short sentences, no idioms, for the Hindi and Bahasa auto-dub.

Every number below is printed on the page, so they will match what the viewer sees. Before recording, open the page fresh and leave it on "None (baseline)" at the last epoch.

## Hook (0:00–0:45)

| Show | Say |
| --- | --- |
| The page on "None (baseline)", chart showing the pink gap | "Look at these two lines. The blue one is how often the model is right on the images it trained on. The orange one is images it has never seen." |
| Point at the pink shaded area | "They should move together. They do not. This pink area is the problem, and it has a name: overfitting." |
| Point at the Gap stat: 7.7 pts | "The model is 7.7 points better on the data it memorized. That part of its skill is fake." |
| Same | "There are four standard ways to fix this. Most courses teach them as four separate topics. Today I want to show you something simpler: they all plug into the same five lines of code." |

## Part 1: The skeleton (0:45–2:00)

Goal: the viewer sees the pipeline before any technique is added.

| Show | Say |
| --- | --- |
| Point at the code panel, top to bottom | "This is every Keras model you will ever write. Three steps. Build, compile, fit." |
| Point at the Sequential block | "Build: a list of layers. 784 pixels come in. Three hidden layers of 256, 128 and 64. Ten outputs, one for each kind of clothing." |
| Point at compile | "Compile: which optimizer, which loss function. Adam, and crossentropy for picking one of ten classes." |
| Point at fit | "Fit: here is the data, train for 30 rounds. That is it. 242,000 numbers get adjusted." |
| Point at the line "Attaches as nothing added" | "Right now nothing is added. This is the plain model, and it overfits." |
| Same | "Watch this line as I click through the four techniques. It tells you where each one plugs in." |

## Part 2: The baseline (2:00–3:15)

| Show | Say |
| --- | --- |
| Drag the epoch slider back to 1 | "Let me go back to the first epoch and play the training." |
| Press Play, let it run | "Watch the two lines. For the first two or three epochs they rise together. The model is learning real patterns." |
| Let it continue, point at where they separate | "Now they split. Blue keeps climbing. Orange goes flat. From here, every extra round of training only helps on data the model already saw." |
| Let it reach epoch 30 | "At the end: 95.3 percent on training data, 87.7 percent on new data. A gap of 7.7 points." |
| Point at the bar chart below, baseline row | "On the final test set it scored 0.870. Remember that number." |

## Part 3: L2, an argument inside the layer (3:15–4:15)

| Show | Say |
| --- | --- |
| Click "L2 (weight decay)" | "First fix. L2, also called weight decay." |
| Point at the green highlighted lines in the code | "Look where it goes. Inside each Dense layer, as an extra argument. No new layers. The model has the same shape." |
| Same | "What it does: it adds a penalty for large weights. The model is allowed to fit the data, but it has to pay for every big number it uses." |
| Press Play | "Watch the two lines now. They stay much closer together." |
| Point at the stats | "At the end: 90.9 percent training, 87.7 percent validation. The gap dropped from 7.7 points to 3.2." |
| Point at the bar chart, L2 row | "But look at the test score. 0.872. Almost the same as before. This is important. L2 did not make the model smarter. It stopped the model from fooling us." |

## Part 4: Dropout, a layer between the layers (4:15–5:30)

| Show | Say |
| --- | --- |
| Click "Dropout" | "Second fix. Dropout." |
| Point at the highlighted Dropout lines | "This one is a real layer. It sits between the Dense layers. Three of them." |
| Same | "What it does: on every minibatch, it switches off 30 percent of the neurons at random. The model can never rely on one neuron, because that neuron might be gone next time." |
| Press Play, point at the start where orange is above blue | "Watch the beginning. The orange line is above the blue one. That looks wrong, but it is not." |
| Same | "Dropout is only on during training. So the model is handicapped on the blue line and running at full strength on the orange line." |
| Let it finish, point at the Gap stat | "At the end: 90.0 and 89.4. A gap of 0.6 points. Almost nothing." |
| Point at the bar chart, Dropout row | "And the test score is 0.888. The best of all five. Dropout wins both." |

## Part 5: Early stopping, a callback on fit (5:30–6:45)

| Show | Say |
| --- | --- |
| Click "Early stopping" | "Third fix, and it is the strangest one, because it changes nothing in the model." |
| Point at the Sequential block, unhighlighted | "Look at the layers. No regularizer. No dropout. This is the plain model from the beginning." |
| Point at the highlighted EarlyStopping block and the callbacks line | "The change is down here. A callback, handed to fit. It watches the validation loss after every epoch." |
| Point at "patience=5" | "Patience 5 means: if the validation loss does not improve for five epochs in a row, stop." |
| Point at "restore_best_weights=True" | "And this line matters. When it stops, it throws away the last five epochs and goes back to the best one." |
| Point at the epoch counter: 11 | "We asked for a maximum of 100 epochs. It stopped itself after 11." |
| Press Play | "The gap never gets a chance to open, because training ends before it can." |
| Point at the bar and stats | "Gap 3.1 points, test score 0.880. Very good, for one extra line of code and a third of the training time." |

## Part 6: Batch normalization, the surprise (6:45–8:00)

| Show | Say |
| --- | --- |
| Click "Batch normalization" | "Fourth. And this one does not do what you expect." |
| Point at the highlighted BatchNormalization lines | "Same position as dropout: a layer after each hidden Dense layer." |
| Same | "What it does: it resets the numbers passing through, back to average zero and spread one. I made a whole video about this one, Episode 3." |
| Press Play | "Now watch. The blue line goes higher and faster than any other model we have seen." |
| Let it finish, point at the Gap stat: 11.1 | "97.4 percent on training. 86.3 on validation. A gap of 11.1 points. That is worse than the model with no fix at all." |
| Point at the bar chart, Batch Norm row | "And the test score is 0.862. The lowest of the five." |
| Same | "Here is the lesson. Batch normalization is not a regularizer. It makes training easier and more stable. And when training is easier, the model memorizes the training set faster. It is a tool for a different job." |

## Wrap-up (8:00–9:00)

| Show | Say |
| --- | --- |
| Scroll to the bar chart, all five rows visible | "Five models. Same 242,000 parameters. Same data. Here is everything." |
| Point at the gap column, top to bottom | "Dropout: 0.6 points. Early stopping: 3.1. L2: 3.2. No fix: 7.7. Batch norm: 11.1." |
| Point at the test numbers | "Now the test scores. 0.888, 0.880, 0.872, 0.870, 0.862." |
| Same | "Look how close they are. From the best to the worst is about two and a half points. None of these techniques made the network much better at recognizing clothing." |
| Same | "What they changed is whether you can trust the training numbers. That is what regularization is for." |
| Point at the code panel | "And remember the one simple idea: build, compile, fit. L2 goes inside a layer. Dropout and batch norm go between layers. Early stopping goes on fit. Four techniques, three places." |
| The repo README | "The page is in the description. Click the buttons, drag the slider, break it. Tell me in the comments which topic you want next." |

## Episode 4 recording checklist

- [ ] Check the live link opens: https://fleabyte26.github.io/Introduction-to-Deep-Learning/regularization-compared.html
- [ ] Reload the page before recording so it starts on "None (baseline)" at the last epoch
- [ ] Browser zoom at 125% so the numbers are readable on a phone
- [ ] Light mode for recording
- [ ] Bookmarks bar hidden, notifications off
- [ ] Practice the Play timing once: a full sweep takes about 5 seconds, so start talking before you press it
- [ ] Have the bar chart and the top panel both reachable with one scroll
- [ ] Publishing on BrokenLabsFix (Hindi/Bahasa audience, auto-dubbing on)

## Episode 4 title options

1. Four Ways to Stop a Neural Network Overfitting (See Them Side by Side)
2. Dropout vs L2 vs Early Stopping vs Batch Norm: One Network, Real Numbers
3. Why Batch Normalization Made My Model Worse

## Episode 4 description

```
One interactive page comparing four regularization techniques on the same neural network, with real training curves. No install, no code.

Four ways to close the overfitting gap
https://fleabyte26.github.io/Introduction-to-Deep-Learning/regularization-compared.html

All pages and source: https://github.com/Fleabyte26/Introduction-to-Deep-Learning

Episode 1 (linear regression, gradient descent, sigmoid): https://www.youtube.com/watch?v=YQF4lMUv9d4
Episode 2 (weight decay, early stopping, dropout, SGD, minibatch size, momentum): https://www.youtube.com/watch?v=_69pXSvmCOY
Episode 3 (batch normalization): https://www.youtube.com/watch?v=r_NqjxmXD8g

Based on Dive into Deep Learning (d2l.ai), sections 3.7, 5.5, 5.6 and 8.5. Trained on Fashion-MNIST: 36,000 training images, 12,000 validation, 12,000 test.

0:00 The gap
0:45 Build, compile, fit
2:00 The baseline overfits
3:15 L2: inside the layer
4:15 Dropout: between the layers
5:30 Early stopping: on fit
6:45 Batch norm goes the wrong way
8:00 What the numbers say
```

## Episode 4 tags

overfitting, regularization, dropout, L2 regularization, weight decay, early stopping, batch normalization, deep learning, neural networks, fashion mnist, keras, tensorflow, d2l, dive into deep learning, machine learning tutorial, validation accuracy, interactive explainer
