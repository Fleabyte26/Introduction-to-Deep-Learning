# Video Script: Deep Learning Without the Formula Wall

BrokenLabsFix · Kris Hornung · November 2026

This file holds two scripts. Episode 1, below, covers the first three pages. [Episode 2](#episode-2-how-models-stop-memorizing-and-learn-faster) covers the six newer pages: weight decay, early stopping, dropout, SGD, minibatch size and momentum.

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
