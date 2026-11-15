# Video Script: Deep Learning Without the Formula Wall

BrokenLabsFix · Kris Hornung · November 2026

Watch the recorded video: https://www.youtube.com/watch?v=YQF4lMUv9d4

This file holds eleven scripts. Episode 1, below, covers the first three pages. [Episode 2](#episode-2-how-models-stop-memorizing-and-learn-faster) covers the six newer pages: weight decay, early stopping, dropout, SGD, minibatch size and momentum. [Episode 3](#episode-3-batch-normalization-you-can-drag) covers the batch normalization page. [Episode 4](#episode-4-four-ways-to-close-the-overfitting-gap) covers the regularization comparison page. [Episode 5](#episode-5-convolutions-channels-and-the-arrow) covers the three convolution pages. [Episode 6](#episode-6-image-augmentation-and-fine-tuning) covers the image augmentation and fine-tuning pages. [Episode 7](#episode-7-three-cnns-five-epochs) covers the three-CNN comparison page. [Episode 8](#episode-8-keras-in-five-moves-and-nine-ways-to-break-it) covers the Keras patterns page. [Episode 9](#episode-9-recurrent-neural-networks) covers the recurrent network page. [Episode 10](#episode-10-lstm-a-memory-with-gates) covers the LSTM page. [Episode 11](#episode-11-attention-do-not-remember-look-back) covers the attention page.

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

# Episode 5: Convolutions, Channels and the Arrow

BrokenLabsFix · Kris Hornung · November 2026

Earlier episodes: [Episode 1](https://www.youtube.com/watch?v=YQF4lMUv9d4) · [Episode 2](https://www.youtube.com/watch?v=_69pXSvmCOY) · [Episode 3](https://www.youtube.com/watch?v=r_NqjxmXD8g) · [Episode 4](https://www.youtube.com/watch?v=b4tLpx3_934)

## Episode 5 at a glance

An 11-minute walkthrough of three pages, in reading order. The thread running through all of them: a convolution layer has only three things to decide, and each page answers one of them. How big is the output, how many channels come out, and how many of these can we stack.

| Part | Time | Page | What happens |
| --- | --- | --- | --- |
| Hook | 0:00–0:50 | — | Three questions |
| Part 1 | 0:50–4:00 | padding-stride | How big is the output |
| Part 2 | 4:00–7:30 | channels | Where sixty-four channels come from |
| Part 3 | 7:30–10:15 | residual | Why we can stack a hundred |
| Wrap-up | 10:15–11:00 | — | The three numbers |

Pages:
- [padding-stride.html](https://fleabyte26.github.io/Introduction-to-Deep-Learning/padding-stride.html)
- [channels.html](https://fleabyte26.github.io/Introduction-to-Deep-Learning/channels.html)
- [residual.html](https://fleabyte26.github.io/Introduction-to-Deep-Learning/residual.html)

Same format: left column is what to do on screen, right column is what to say. Short sentences for the Hindi and Bahasa auto-dub.

Every number below is printed on the pages, so reload each one fresh before recording and follow the clicks in order.

## Hook (0:00–0:50)

| Show | Say |
| --- | --- |
| The padding and stride page, on the default view | "A convolution layer looks simple. A small window slides over an image. But when you write one, the library asks you three questions." |
| Point at the sliders | "How big is the output. How many channels come out. And the one nobody asks out loud: how many of these can I stack before it stops working." |
| Same | "Three pages today, one for each question. All three are free, all three run in your browser. Links in the description." |
| Same | "This is chapter 7 and chapter 8 of Dive into Deep Learning. No formulas to memorize. Let us go." |

## Part 1: How big is the output (0:50–4:00)

Page: padding-stride.html. Reload it first so it opens on input 8, kernel 3, padding 0, stride 1.

| Show | Say |
| --- | --- |
| Point at the input grid, then the output grid | "On the left, an 8 by 8 image. On the right, the output. It is 6 by 6. The image got smaller, and nobody asked it to." |
| Press Play, watch the window step across | "Here is why. The window is 3 by 3. It has to fit inside the image. So it can only sit in 6 positions across, and 6 down." |
| Point at the formula panel | "The page writes it out. Room is 5. Steps is 5. Output is 6. That is the whole formula." |
| Same | "Room is how much space the window has to move in. Steps is how many moves fit. Output is steps plus one, because the first position counts too." |
| Click "Same size" | "Now watch this. I add one ring of padding. Zeros around the edge." |
| Point at the purple cells, then the output | "The image is now 10 by 10 with the zeros. The output is 8 by 8. Same as the original image." |
| Same | "This is why you see 3 by 3 with padding 1 everywhere in real networks. It is the pair that keeps the size." |
| Click "Big kernel 5x5" | "And it generalizes. A 5 by 5 kernel with padding 2. Output is still 8 by 8." |
| Click "Halve it" | "Now stride. The window jumps 2 instead of 1. Output is 4 by 4. Half the size, on purpose." |
| Same | "This is how a network goes from a 224 pixel photo down to a 7 by 7 feature map. Not by shrinking slowly. By halving, a few times." |
| Click "Cells get dropped" | "One warning before we move on. Stride 3 here." |
| Point at the red cells | "The red cells are never visited. The window jumped past them. Those are real pixels, and the network never sees them." |
| Point at the verdict box | "Nothing warns you about this. No error, no message. The page is telling you because your code will not." |

## Part 2: Where sixty-four channels come from (4:00–7:30)

Page: channels.html. Reload so it opens on 3 input channels, 4 output channels, 3 by 3.

| Show | Say |
| --- | --- |
| Point at the three colored squares on the left | "A color photo has three channels. Red, green, blue. Three layers of numbers." |
| Point at the output square | "The first layer of a real network turns that into sixty-four channels. So two questions. Where do sixty-four come from, and what happened to the three?" |
| Point at the three kernel squares in the middle | "Here is the answer, and most diagrams get it wrong. The kernel is not a square. It is a stack. One 3 by 3 slice for every input channel." |
| Trace the three curves converging | "Each slice slides over its own channel. Then the three results are added together. Added, pixel by pixel." |
| Point at the single output square | "Three channels went in. One plane came out. That is rule one: input channels are summed away." |
| Click "Next output channel" twice | "So where does the second output channel come from? A second stack, with its own weights. And a third. And a fourth." |
| Point at the strip of four squares | "Four output channels means four complete stacks. Sixty-four means sixty-four. That is rule two: output channels are a choice you make." |
| Same | "The input count is not a choice. The previous layer decided it. The output count is the dial you turn." |
| Point at the formula panel | "Now the cost. Kernel shape is 3, 3, 3, 4. Multiply them: 108 weights, plus 4 biases. 112 numbers." |
| Click "1x1 channel mixer" | "One special case worth knowing. A 1 by 1 kernel." |
| Point at the verdict | "There is no sliding. No neighborhood. Each output pixel is just a mix of the channels at that same spot. It changes the channel count, and nothing else." |
| Scroll to the parameter table | "And here is why this matters. Real numbers from real networks." |
| Point at row 1 | "First layer of VGG: 3 channels in, 64 out. 1,792 parameters. Small." |
| Point at row 2 | "A deep layer of VGG: 512 in, 512 out. Two point three million parameters. In one layer." |
| Same | "Both channel counts are in the multiplication. Double them both and the layer gets four times bigger. That is the whole story of why deep networks got expensive." |
| Point at row 3 | "And that is why the 1 by 1 trick matters. 64 to 256 channels for sixteen thousand parameters. Cheap. Remember it for the next part." |

## Part 3: Why we can stack a hundred (7:30–10:15)

Page: residual.html.

| Show | Say |
| --- | --- |
| The page, reading the opening line | "2015. Someone trained a 56 layer network and a 20 layer network. The deep one was worse on the training data." |
| Same | "Not overfitting. Worse at the data it was looking at. And that should be impossible, because a deep network can always copy a shallow one and do nothing with the extra layers." |
| Click "Plain block" | "Here is a normal block. Two convolutions, two batch norms, activations. The input goes in the left, comes out the right." |
| Point at the line under the diagram | "Now ask it to do nothing. To pass the input through unchanged. Look what it needs: f of x equals x. Every weight exactly right, to copy the input." |
| Same | "That is a hard thing to learn. And a block that gets it almost right damages the signal a little. Stack 36 of those and it adds up." |
| Click "Residual block" | "Now the fix. One arrow." |
| Trace the green arrow | "The input skips over the whole block and gets added to the output at the end. That is it. That is the entire idea." |
| Point at the line under the diagram | "Now ask it to do nothing again. It needs f of x equals zero. Push the weights toward zero." |
| Same | "Gradient descent is very good at pushing weights toward zero. It is the easiest thing it does. So the extra blocks can sit out quietly until they find something useful to add." |
| Scroll to the gradient chart | "There is a second reason, and you can watch it." |
| Point at the red line | "When the network learns, the signal travels backwards from the output to the first layer. Every block multiplies it. This slider says each block multiplies by 0.8." |
| Drag Depth to 60 | "Sixty blocks. Point eight, multiplied by itself, sixty times." |
| Point at the two stat boxes | "Through the layers: 1.9 times ten to the minus six. That is a millionth. The first block has stopped learning." |
| Point at the green line | "Through the skip connections: one. Exactly one. Because the arrow multiplies by one, every single time." |
| Drag Depth back and forth | "Watch the red line move as I change the depth. The green line never moves. That is what the arrow buys you." |
| Scroll to "The practical details" | "Two details for when you write one. If the block changes the channel count, the arrow needs a 1 by 1 convolution to match the shapes. That is the cheap trick from the last page." |
| Same | "And ResNet-152 was eight times deeper than VGG, with fewer parameters. The same arrow is in almost every architecture since. Transformers too." |

## Wrap-up (10:15–11:00)

| Show | Say |
| --- | --- |
| The padding page briefly | "Three questions, three answers." |
| Same | "How big is the output? Room, steps, plus one. Padding keeps the size, stride cuts it." |
| The channels page briefly | "How many channels? Input channels get summed away. Output channels are a stack each, and both counts multiply into the parameter cost." |
| The residual page briefly | "How deep can we go? As deep as you like, if every block has an arrow around it." |
| The repo README | "All three pages are in the description. Open them, drag everything, break them. That is what they are for." |
| Same | "Tell me in the comments what you want next. See you in the next one." |

## Episode 5 recording checklist

- [ ] Check all three links open before recording
- [ ] Reload each page as you reach it, so it starts at its default values
- [ ] Browser zoom at 125% so the numbers are readable on a phone
- [ ] Light mode, bookmarks bar hidden, notifications off
- [ ] Have the three pages in three tabs, in order, so the switches are clean
- [ ] Dry run Part 1: it has the most clicks, and each preset resets the animation
- [ ] On the residual page, drag the depth slider slowly at the end; the contrast between the moving red line and the flat green line is the shot that matters
- [ ] Publishing on BrokenLabsFix (Hindi/Bahasa audience, auto-dubbing on)

## Episode 5 title options

1. How Convolutions Actually Work: Size, Channels, and Depth
2. The Three Questions Every Convolution Layer Asks
3. Why One Arrow Let Neural Networks Go 150 Layers Deep

## Episode 5 description

```
Three free interactive pages covering how a convolution layer works: output size, channels, and residual connections. No install, no code.

Padding and stride you can drag
https://fleabyte26.github.io/Introduction-to-Deep-Learning/padding-stride.html

Where the channels go
https://fleabyte26.github.io/Introduction-to-Deep-Learning/channels.html

The arrow that made networks deep
https://fleabyte26.github.io/Introduction-to-Deep-Learning/residual.html

All pages and source: https://github.com/Fleabyte26/Introduction-to-Deep-Learning

Episode 1 (linear regression, gradient descent, sigmoid): https://www.youtube.com/watch?v=YQF4lMUv9d4
Episode 2 (weight decay, early stopping, dropout, SGD, minibatch size, momentum): https://www.youtube.com/watch?v=_69pXSvmCOY
Episode 3 (batch normalization): https://www.youtube.com/watch?v=r_NqjxmXD8g
Episode 4 (comparing four regularization techniques): https://www.youtube.com/watch?v=b4tLpx3_934

Based on Dive into Deep Learning (d2l.ai), sections 7.1 to 7.4 and 8.6.

0:00 Three questions
0:50 How big is the output
4:00 Where sixty-four channels come from
7:30 Why we can stack a hundred
10:15 The three answers
```

## Episode 5 tags

convolution, convolutional neural network, CNN, padding, stride, kernel size, channels, feature maps, 1x1 convolution, residual connection, skip connection, ResNet, VGG, vanishing gradient, deep learning, d2l, dive into deep learning, keras, machine learning tutorial, interactive explainer

# Episode 6: Image Augmentation and Fine-Tuning

BrokenLabsFix · Kris Hornung · November 2026

Watch the recorded video: https://www.youtube.com/watch?v=IIWBWRTOFEg

Earlier episodes: [Episode 1](https://www.youtube.com/watch?v=YQF4lMUv9d4) · [Episode 2](https://www.youtube.com/watch?v=_69pXSvmCOY) · [Episode 3](https://www.youtube.com/watch?v=r_NqjxmXD8g) · [Episode 4](https://www.youtube.com/watch?v=b4tLpx3_934) · [Episode 5](https://www.youtube.com/watch?v=8Ab74b_z-Es)

## Episode 6 at a glance

A 9-minute walkthrough of two pages. The thread: a network learns whatever is in its pictures, including the accidents. Augmentation removes the accidents; fine-tuning lets you start from a network that has already seen ten million pictures.

This episode was made in a different order from the first five: the pages first, then this script, then the video, which was rendered directly from the pages with a synthetic narration voice. The timings below are the rendered video's real timings. To re-record it in your own voice, follow the Show column as a click list.

| Part | Time | Page | What happens |
| --- | --- | --- | --- |
| Hook | 0:00–0:51 | image-augmentation | Zero percent of mirrored boots |
| Part 1 | 0:51–2:25 | image-augmentation | Flip, crop, brightness, hue |
| Part 2 | 2:25–3:01 | image-augmentation | Never the same pixels twice |
| Part 3 | 3:01–3:39 | image-augmentation | The label must survive |
| Part 4 | 3:39–5:00 | image-augmentation | Each augmentation buys one kind of robustness |
| Part 5 | 5:00–6:33 | fine-tuning | Copy everything except the last layer |
| Part 6 | 6:33–7:27 | fine-tuning | One rate cannot suit both |
| Part 7 | 7:27–8:22 | fine-tuning | Scratch, fine-tune, or freeze |
| Wrap-up | 8:22–8:50 | — | Two ideas |

Pages:
- [image-augmentation.html](https://fleabyte26.github.io/Introduction-to-Deep-Learning/image-augmentation.html)
- [fine-tuning.html](https://fleabyte26.github.io/Introduction-to-Deep-Learning/fine-tuning.html)

Same format: left column is what to do on screen, right column is what to say. Short sentences for the Hindi and Bahasa auto-dub.

Every number below is printed on the pages. The page draws its random numbers from a fixed seed, so after a reload (or the Reset button) the three "Roll the dice" presses and the "Next epoch" presses always give the same pictures.

## Episode 6 hook (0:00–0:51)

Page: image-augmentation.html.

| Show | Say |
| --- | --- |
| The augmentation page, scrolled to "What it does to a real network", with "No augmentation" selected | "This is a real neural network. It was trained to recognise clothing. On normal test pictures, it gets 83 percent right." |
| Point at the verdict box | "It recognises 92 percent of ankle boots. Now mirror the boots, so the toe points right instead of left. It recognises zero percent. Not one." |
| Point at the three small boot pictures | "Same boots. Same network. The only change is the direction. So what did this network learn? Not what a boot is. It learned where the toe goes." |
| Scroll to the top of the page | "Today, two pages. The first one fixes this problem with a few lines of code. It is called image augmentation. The second one shows how to borrow a network that someone else already trained. That is called fine-tuning." |
| Same | "This is chapter 14 of Dive into Deep Learning, sections 14.1 and 14.2. Both pages are free, and they run in your browser. The links are in the description." |

## Episode 6 Part 1: One picture, changed (0:51–2:25)

Page: image-augmentation.html.

| Show | Say |
| --- | --- |
| The first panel. Press Reset. Point at the left picture | "Here is one training picture. A boot, pointing left. Without augmentation, the network sees exactly these pixels, in every epoch." |
| Click the "Flip left-right" preset | "First change. Flip it left to right. The toe now points right. Every pixel moved. But the label did not change. It is still a boot." |
| Point at the verdict box | "This is the oldest trick in image augmentation, and the cheapest. In the code, it is one line." |
| Click "Flip up-down" | "Now flip it upside down. It is still a boot. But no camera will ever send the network a picture like this. So the book says up-down flips are much less common." |
| Click "Crop". Point at the dashed box, then the right picture | "Next, a random crop. The dashed box is the part we keep. Then it is stretched back to full size. The boot is now bigger, and it sits in a different place." |
| Point at the numbers panel | "The page shows the numbers. This crop keeps 40 percent of the area. In the book, the crop is random: anywhere from 10 percent to 100 percent of the picture." |
| Click "Brightness" | "Now colour. First, brightness. Every pixel is multiplied by 1.4. The room got brighter. The boot is still a boot." |
| Click "Hue" | "And hue. The colours rotate around the colour wheel. The red boot is now a green boot. A red boot and a green boot are both boots, so the network should not care about the colour." |
| Click "All together" | "In practice, you chain them. Flip, crop, brightness and hue, all at once." |
| Press Reset, then "Roll the dice" | "And you do not choose the numbers. The computer does, at random. Press Roll the dice." |
| Press "Roll the dice" again | "Again. A different crop, a different light, a different colour." |
| Press it a third time. Point at the last line of the numbers panel | "And again. Every press is a new picture. And the last line never changes. Label: boot." |

## Episode 6 Part 2: A new version every epoch (2:25–3:01)

Page: image-augmentation.html.

| Show | Say |
| --- | --- |
| Scroll to "One image, a new version every epoch" | "Here is what that means for training. Eight loads of the same picture. Eight different inputs." |
| Same | "Notice that nothing new is saved on disk. The dataset is not bigger. The change happens each time a picture is loaded." |
| Press "Next epoch" | "Next epoch. The same boot again, and eight new versions." |
| Press "Next epoch" again. Point at the line under the button | "And again. After three epochs, the network has seen this boot three times, and it has never seen the same pixels twice. That is why it cannot simply memorise the picture." |
| Point at the tightest crops in the grid | "One warning. Look at the tightest crops. In some of them, you can hardly tell it is a boot. A very aggressive crop can cut the subject out of the picture." |

## Episode 6 Part 3: The one rule (3:01–3:39)

Page: image-augmentation.html.

| Show | Say |
| --- | --- |
| Scroll to "The one rule". "A boot" is selected | "Now, the one rule. An augmentation is only allowed if the answer is still the same afterwards. For a boot, a left-right flip is safe." |
| Click "The letter b" | "Now the letter b. Flip it left to right, and it is a d. Flip it upside down, and it is a p. Both flips, and it is a q." |
| Point at the verdict box | "If you augment handwriting with flips, you show the network a d, and tell it the answer is b. The same operation that was safe for the boot is now teaching mistakes." |
| Click "The digit 6" | "Same with digits. Flip a 6 both ways, and it is a 9. So the rule is not about the operation. It is about the subject. Always ask: is the label still true?" |

## Episode 6 Part 4: What it does to a real network (3:39–5:00)

Page: image-augmentation.html.

| Show | Say |
| --- | --- |
| Scroll to "What it does to a real network". Click "No augmentation" | "Back to the real network. One thousand training pictures from Fashion-MNIST. In this dataset, every shoe points left, and every item sits in the centre." |
| Point at the four bars | "No augmentation. 100 percent on the training pictures. 83.4 on normal test pictures. But 60 percent when mirrored. And 57 percent when the picture moves just two pixels." |
| Click "Random left-right flip" | "Now add one line: a random left-right flip. Mirrored pictures jump to 82.7 percent. Mirrored ankle boots go from zero to 92 percent." |
| Point at the verdict box | "But look at the shifted test. Still broken, at 51 percent. A flip teaches nothing about position." |
| Click "Random shift (crop)" | "Now the other one: a random shift. Shifted pictures recover to 78 percent. Mirrored pictures stay broken. Each augmentation buys exactly the robustness it shows the network, and nothing else." |
| Click "Flip and shift together" | "So chain them. Flip and shift together. 82 percent mirrored. 79 percent shifted. Both holes are closed." |
| Point at the table, first and last rows | "And look at the cost. The normal test score hardly moved. Training accuracy fell from 100 to 91 percent, because the network cannot memorise pictures that keep changing. A lower training score, and a more honest network." |
| Scroll to "Two details the code depends on" | "One last rule before we move on. Augmentation is for training only. At test time there is no randomness. You want one clear answer for each picture." |

## Episode 6 Part 5: Fine-tuning in four steps (5:00–6:33)

Page: fine-tuning.html.

| Show | Say |
| --- | --- |
| Open the fine-tuning page | "Second page. Fine-tuning. Here is the problem. You want to recognise hot dogs, and you have two thousand photos. That is not enough to train a big network from nothing." |
| Same | "But ImageNet has more than ten million photos. And a network trained on ImageNet already knows edges, textures and shapes. Fine-tuning borrows that knowledge. It takes four steps." |
| The first panel, on "Start" | "Start. This is ResNet 18. It has 11.7 million weights. Right now, they are random. The picture goes in at the bottom. The answer comes out at the top." |
| Click "1 · Pretrain" (or press Next step) | "Step one. Someone trains it on ImageNet. This is the expensive part. You do not do it. You download the result." |
| Click "2 · Copy the layers" | "Step two. Build a new model with the same design, and copy every layer across. Every layer except one. The output layer is not copied." |
| Click "3 · New output layer" | "Step three. Why not? Because the old output layer answers the wrong question. It chooses between one thousand ImageNet classes. You have two: hot dog, or not hot dog. So you add a new output layer, with random weights." |
| Point at the numbers panel | "Now the numbers. Copied: 11.1 million weights. New and random: 1,026 weights. That is nine thousandths of one percent of the model. Almost everything is borrowed." |
| Drag "Classes in your own dataset" to 100 | "Even with one hundred classes, the new layer is 51,300 weights. Less than half of one percent." |
| Drag it back to 2. Click "4 · Train on your data" | "Step four. Train on your own pictures. But with two different learning rates. A small rate for the copied layers. Ten times that for the new output layer." |

## Episode 6 Part 6: Why two learning rates (6:33–7:27)

Page: fine-tuning.html.

| Show | Say |
| --- | --- |
| Scroll to "Why two learning rates". Click "One small rate" | "Why two rates? Here is a small cartoon with two numbers. The blue line is the copied layers. They start close to their best value. The orange line is the new layer. It starts far away." |
| Point at the two result boxes | "First, one small learning rate for everything. The copied layers are fine. But after 30 steps, the new layer has learned only 26 percent. Too slow." |
| Click "One big rate" | "So use one big rate. Now the new layer learns 97 percent. But look at the blue line. The copied layers were thrown out of their good position. They started 0.3 from their best value. They ended 68 away." |
| Point at the red verdict box | "The borrowed knowledge is destroyed. This is the classic fine-tuning mistake: a learning rate that is too large." |
| Click "Small, and 10× for the new layer" | "Now the book's recipe. A small rate, and ten times that for the new layer. The copied layers stay safe. The new layer learns 96 percent. Both are satisfied." |

## Episode 6 Part 7: Three ways, and the book's experiment (7:27–8:22)

Page: fine-tuning.html.

| Show | Say |
| --- | --- |
| Scroll to "Three ways to use a pretrained network" | "So there are three ways to use a pretrained network. Train from scratch: everything starts random. Fine-tune: only the output layer starts random, and you train everything, gently." |
| Point at the last row | "And the cheap option: freeze the copied layers, and train only the new layer. 1,026 weights. Try this first when your dataset is very small." |
| Scroll to "The book's experiment" | "The book tests this on hot dogs. One thousand photos of each class for training. Five epochs. The fine-tuned network beats the same network trained from scratch, because it starts from better values." |
| Point at "The detail that bites" | "One detail that bites. The copied layers expect pictures prepared exactly like ImageNet pictures: the same size, and the same colour scaling. That is why every fine-tuning script has six strange numbers at the top." |
| Point at "Where augmentation comes back" | "And notice where augmentation comes back. Training pictures get a random crop and a random flip. Test pictures get no randomness. Random for training, fixed for testing." |

## Episode 6 wrap-up (8:22–8:50)

| Show | Say |
| --- | --- |
| The augmentation page, "All together" preset | "Two ideas today. Image augmentation: change every training picture a little, at random, so the network learns the object, and not the accident. But only in ways that keep the label true." |
| The fine-tuning page, on step 4 | "Fine-tuning: copy a trained network, replace the last layer, and train gently, with a larger rate for the new layer." |
| Same | "Both pages are in the description. Open them, press everything, break them. Tell me in the comments what you want next. See you in the next one." |

## Episode 6 recording checklist

- [ ] Check both live links open before recording
- [ ] Reload each page as you reach it, so the dice and the epochs start from their fixed seed
- [ ] Browser zoom at 125% so the numbers are readable on a phone
- [ ] Light mode, bookmarks bar hidden, notifications off
- [ ] On the augmentation page, press Reset before the three "Roll the dice" presses
- [ ] On the fine-tuning page, drag the class slider back to 2 before clicking step 4
- [ ] Publishing on BrokenLabsFix (Hindi/Bahasa audience, auto-dubbing on)

## Episode 6 title options

1. Image Augmentation and Fine-Tuning Explained | Deep Learning Without the Formula Wall Ep. 6
2. Why This Network Scores 0% on Mirrored Boots | Deep Learning Without the Formula Wall Ep. 6
3. Borrow a Trained Network: Fine-Tuning in Four Steps | Deep Learning Without the Formula Wall Ep. 6

## Episode 6 description

```
Two free interactive pages on image augmentation and fine-tuning. See one picture become many, watch a real network fail on mirrored boots, and step through how a pretrained network is reused. No install, no code.

Image augmentation: one picture becomes many
https://fleabyte26.github.io/Introduction-to-Deep-Learning/image-augmentation.html

Fine-tuning: borrow a network, replace its last layer
https://fleabyte26.github.io/Introduction-to-Deep-Learning/fine-tuning.html

All pages and source: https://github.com/Fleabyte26/Introduction-to-Deep-Learning

Episode 1 (linear regression, gradient descent, sigmoid): https://www.youtube.com/watch?v=YQF4lMUv9d4
Episode 2 (weight decay, early stopping, dropout, SGD, minibatch size, momentum): https://www.youtube.com/watch?v=_69pXSvmCOY
Episode 3 (batch normalization): https://www.youtube.com/watch?v=r_NqjxmXD8g
Episode 4 (comparing four regularization techniques): https://www.youtube.com/watch?v=b4tLpx3_934
Episode 5 (convolutions, channels, residual connections): https://www.youtube.com/watch?v=8Ab74b_z-Es

Based on Dive into Deep Learning (d2l.ai), sections 14.1 and 14.2. The experiment uses Fashion-MNIST: 1,000 training pictures, 3,000 test pictures, average of three runs.

0:00 A network that cannot see mirrored boots
0:51 Flip, crop, brightness, hue
2:25 A new version every epoch
3:01 The one rule: the label must survive
3:39 What augmentation does to a real network
5:00 Fine-tuning in four steps
6:33 Why two learning rates
7:27 Three ways to use a pretrained network
8:22 Two ideas
```

## Episode 6 tags

image augmentation, data augmentation, random crop, horizontal flip, color jitter, fine-tuning, transfer learning, pretrained model, ResNet-18, ImageNet, learning rate, Fashion-MNIST, computer vision, convolutional neural network, deep learning, d2l, dive into deep learning, machine learning tutorial, interactive explainer

# Episode 7: Three CNNs, Five Epochs

BrokenLabsFix · Kris Hornung · November 2026

Watch the recorded video: https://www.youtube.com/watch?v=02nDUhqnNik

Earlier episodes: [Episode 1](https://www.youtube.com/watch?v=YQF4lMUv9d4) · [Episode 2](https://www.youtube.com/watch?v=_69pXSvmCOY) · [Episode 3](https://www.youtube.com/watch?v=r_NqjxmXD8g) · [Episode 4](https://www.youtube.com/watch?v=b4tLpx3_934) · [Episode 5](https://www.youtube.com/watch?v=8Ab74b_z-Es) · [Episode 6](https://www.youtube.com/watch?v=IIWBWRTOFEg)

## Episode 7 at a glance

A 5-minute walkthrough of one page. Three versions of one small CNN were trained on CIFAR-10 for five epochs: plain, with augmentation, and with batch normalization and dropout. The plain one won. The episode uses the real numbers from that run to show why, and why five epochs is too short to judge.

Made the same way as Episode 6: the page first, then this script, then the video rendered from the page with a synthetic narration voice. The timings are the rendered video's real timings.

This episode explains results and concepts only. It does not show or walk through any course assignment or its code.

| Part | Time | What happens |
| --- | --- | --- |
| Hook | 0:00–0:46 | The improved networks lost |
| Part 1 | 0:46–2:02 | Three models, five epochs |
| Part 2 | 2:02–2:58 | Where the plain network hits the wall |
| Part 3 | 2:58–3:40 | 93 percent in one layer |
| Part 4 | 3:40–4:29 | When unsure, it says frog |
| Part 5 | 4:29–5:03 | What five epochs does not say |
| Part 6 | 5:03–5:24 | Small filters, soft patches |
| Wrap-up | 5:24–5:50 | Three lessons |

Page: [cnn-three-models.html](https://fleabyte26.github.io/Introduction-to-Deep-Learning/cnn-three-models.html)

Same format: left column is what to do on screen, right column is what to say. Every number is printed on the page.

## Episode 7 hook (0:00–0:46)

| Show | Say |
| --- | --- |
| The page, first panel, "All three" selected, epoch 5 | "Three neural networks. Same design, same pictures, same five epochs of training. One is plain. One uses image augmentation. One uses batch normalization and dropout." |
| Point at the three lines | "The two with the improvements lost. The plain network scored 60 percent. The other two scored 44 and 48." |
| Same | "So are these techniques useless? No. Today we use real numbers from a real training run to see what happened. And why this result is honest, but would not last." |
| Scroll to the top of the page | "The data is CIFAR-10. Small colour pictures, 32 by 32 pixels, in ten classes: airplanes, cars, birds, cats, and so on. Thirty thousand pictures for training. Ten thousand for validation. Ten thousand for the final test." |

## Episode 7 Part 1: The race (0:46–2:02)

| Show | Say |
| --- | --- |
| Press Play on the first panel and let it run to epoch 5 | "Here is the race. Solid lines are validation accuracy: pictures the network is not trained on. Watch the five epochs." |
| Click "Plain" | "First, the plain network alone. The dashed line is training accuracy. It starts below validation. At epoch four, it crosses above." |
| Point at the verdict box | "That crossing matters. From here, the network does better on pictures it has seen than on new pictures. It has started to memorise." |
| Click "Augmented" | "Now the augmented network. Every training picture is rotated, shifted, and sometimes mirrored. Differently each time. Harder pictures, so slower learning." |
| Point at the dip at epoch 4 | "Look at epoch four. Validation fell from 42 percent to 35 percent. Then it came back to 44. Nothing broke. With augmentation, validation is jumpy. Never trust one epoch alone." |
| Click "Batch norm + dropout" | "Third, batch normalization and dropout. Look how far training is below validation. 36 percent against 48." |
| Point at the verdict box | "That is dropout. During training, half of the dense layer is switched off at random. The network trains with part of itself missing. At validation, everything is switched on." |
| Point at epoch 1 of the solid line | "And the first epoch: validation was only 25 percent. That is batch normalization settling in. One epoch later, 36." |

## Episode 7 Part 2: Fifteen epochs by accident (2:02–2:58)

| Show | Say |
| --- | --- |
| Scroll to "The plain network, left running by accident". Drag the epoch slider to 5 | "Now a confession. In a notebook, if you run a training cell twice, it does not start again. It continues. I ran the plain network's cell three times, by mistake. So it trained for fifteen epochs." |
| Point at the accuracy chart | "Here are the first five. Both lines rise together. This is the part the race measured." |
| Drag the slider slowly to epoch 8 | "Keep going. The gap opens. Training keeps climbing. Validation slows down. At epoch eight, validation loss reaches its lowest point: 1.12." |
| Drag on to epoch 15 | "And then it turns. Validation loss goes up, while training loss keeps going down." |
| Point at the numbers panel | "At epoch fifteen: 79 percent on its own pictures. 62 percent on new ones. A gap of seventeen points. The network is learning facts about its training pictures that are not true of other pictures." |
| Point at the red verdict box | "This is overfitting. And this is the wall that augmentation and dropout are built to move. Five epochs is too short to reach the wall." |

## Episode 7 Part 3: Where the weights are (2:58–3:40)

| Show | Say |
| --- | --- |
| Scroll to "Where the 282,250 weights are" | "What is inside this network? 282,250 weights. Here is every layer." |
| Click the first convolution row | "The first convolution. 32 filters. Each filter looks at 3 by 3 pixels, in 3 colours. 27 weights and one bias. Times 32 filters: 896." |
| Click the "Dense, 64 units" row | "Now look at the bars. The dense layer holds 262,000 weights. That is 93 percent of the model. The convolution layers, the part that makes this a CNN, are tiny." |
| Click "With batch norm and dropout" | "Add batch normalization and dropout. The total goes up by 384 weights. So the third network is not bigger. It is the same network, with brakes." |

## Episode 7 Part 4: What it confuses (3:40–4:29)

| Show | Say |
| --- | --- |
| Scroll to "What the third network confuses" | "What does a network like this get wrong? This is a confusion matrix. Each row is the truth. Each column is what the network said. The diagonal is correct answers." |
| "Automobile called truck" is selected. Point at the text on the right | "The biggest mistake: 416 automobiles were called truck. Only 413 were called automobile. Both are boxes on wheels." |
| Click "Cat called frog" | "Now look at the frog column. The network said frog 2,179 times. There are only one thousand frogs. Cats, deer, birds, dogs: when it is not sure, it says frog." |
| Click "Frog" | "So frog has the best score on the board. 836 out of one thousand. But part of that is an illusion. If you answer frog all the time, you will catch most of the frogs." |
| Click "Cat called dog" | "And the classic one. A cat is called a dog more often than it is called a cat." |

## Episode 7 Part 5: The final test (4:29–5:03)

| Show | Say |
| --- | --- |
| Scroll to "The final test, and what it does not say" | "The final test. Ten thousand pictures no network has seen. Plain: 59.6 percent. Batch norm and dropout: 47.7. Augmented: 44." |
| Point at the second set of bars | "But look at the second chart. Training accuracy minus validation accuracy. The plain network is already above zero. It has started to memorise. The other two have not." |
| Point at the verdict box | "Augmentation and dropout make learning harder, on purpose. Five epochs measures the cost, and none of the benefit. A fair test would train all three until validation stops improving." |

## Episode 7 Part 6: What the first layer learned (5:03–5:24)

| Show | Say |
| --- | --- |
| Scroll to "What the first layer learned" | "One last look. These are eight filters from the first layer of the plain network. Each is 3 by 3 pixels." |
| Click one of the filters | "Textbooks show sharp edge detectors. These are soft colour patches. That is normal for tiny filters after five epochs. Light on one side and dark on the other is still an edge detector. Just a very small one." |

## Episode 7 wrap-up (5:24–5:50)

| Show | Say |
| --- | --- |
| Back to the first panel, "All three" | "So, three lessons. One: more techniques do not mean a better score. Not in five epochs." |
| Same | "Two: watch the gap between training and validation. It tells you more than the score." |
| Same | "Three: one number from one epoch is noisy. Look at the whole curve." |
| Same | "The page is in the description. Every number on it is from my own run. Press Play, click the squares, and tell me in the comments what to build next. See you in the next one." |

## Episode 7 recording checklist

- [ ] Check the live link opens
- [ ] Browser zoom at 125%, light mode, bookmarks bar hidden, notifications off
- [ ] On the first panel use Play for the race; on the second, drag the slider by hand so you control the pace
- [ ] Publishing on BrokenLabsFix (Hindi/Bahasa audience, auto-dubbing on)

## Episode 7 title options

1. Why My Plain CNN Beat Dropout and Augmentation | Deep Learning Without the Formula Wall Ep. 7
2. Three CNNs, Five Epochs, Real Numbers: Overfitting Explained | Deep Learning Without the Formula Wall Ep. 7
3. CNN on CIFAR-10: Training Curves and Confusion Matrix Explained | Deep Learning Without the Formula Wall Ep. 7

## Episode 7 description

```
One free interactive page built from a real training run: three versions of a small CNN on CIFAR-10, five epochs each. Watch the race, see a real overfitting curve, count where the weights are, and click through a confusion matrix. No install, no code.

Three CNNs, five epochs: why the plain one won
https://fleabyte26.github.io/Introduction-to-Deep-Learning/cnn-three-models.html

All pages and source: https://github.com/Fleabyte26/Introduction-to-Deep-Learning

Episode 1 (linear regression, gradient descent, sigmoid): https://www.youtube.com/watch?v=YQF4lMUv9d4
Episode 2 (weight decay, early stopping, dropout, SGD, minibatch size, momentum): https://www.youtube.com/watch?v=_69pXSvmCOY
Episode 3 (batch normalization): https://www.youtube.com/watch?v=r_NqjxmXD8g
Episode 4 (comparing four regularization techniques): https://www.youtube.com/watch?v=b4tLpx3_934
Episode 5 (convolutions, channels, residual connections): https://www.youtube.com/watch?v=8Ab74b_z-Es
Episode 6 (image augmentation and fine-tuning): https://www.youtube.com/watch?v=IIWBWRTOFEg

Concepts from Dive into Deep Learning (d2l.ai), chapters 7, 8 and 14. Dataset: CIFAR-10, 30,000 training pictures, 10,000 validation, 10,000 test.

0:00 The improved networks lost
0:46 The race: three models, five epochs
2:02 Fifteen epochs by accident
2:58 Where the 282,250 weights are
3:40 What the network confuses
4:29 The final test
5:03 What the first layer learned
5:24 Three lessons
```

## Episode 7 tags

convolutional neural network, CNN, CIFAR-10, overfitting, training curves, validation accuracy, confusion matrix, dropout, batch normalization, data augmentation, early stopping, Keras, deep learning, d2l, dive into deep learning, machine learning tutorial, interactive explainer

# Episode 8: Keras in Five Moves, and Nine Ways to Break It

BrokenLabsFix · Kris Hornung · November 2026

Watch the recorded video: https://www.youtube.com/watch?v=75z6xEjq__Y

## Episode 8 at a glance

A 4-minute walkthrough of one page about code, not results. First the pattern every Keras program follows: build, compile, fit, evaluate, predict. Then nine real bugs from one afternoon of typing code by hand, each with the exact message Python gives.

Made the same way as Episodes 6 and 7: the page first, then this script, then the video rendered from the page with a synthetic narration voice. The model on the page is a small example made for it, not an assignment solution.

| Part | Time | What happens |
| --- | --- | --- |
| Hook | 0:00–0:27 | One wrong character |
| Part 1 | 0:27–1:39 | Build, compile, fit, evaluate, predict |
| Part 2 | 1:39–2:13 | Every mark has a job |
| Part 3 | 2:13–3:47 | Real mistakes, real error messages |
| Part 4 | 3:47–4:08 | Copy, read the last line, restart and run all |
| Wrap-up | 4:08–4:18 | Five moves, nine bugs, three habits |

Page: [keras-patterns.html](https://fleabyte26.github.io/Introduction-to-Deep-Learning/keras-patterns.html)

Same format: left column is what to do on screen, right column is what to say.

## Episode 8 hook (0:00–0:27)

| Show | Say |
| --- | --- |
| The page, scrolled to "Spot the bug", bug 4 selected | "This code has one wrong character. Just one. Can you see it?" |
| Click the MaxPooling2D line | "Here it is. A period, where a comma should be. And look at what Python says. MaxPooling2D object has no attribute Flatten. That message does not say comma anywhere." |
| Scroll to the top of the page | "Today is about Keras code. First, the pattern. Every Keras program has the same five moves. Then, nine real bugs. I made every one of them in a single afternoon, typing code by hand." |

## Episode 8 Part 1: The five moves (0:27–1:39)

| Show | Say |
| --- | --- |
| The first panel. Click "Build" | "Move one: build. You give Sequential a list of layers, in the order the picture passes through them. Top to bottom." |
| Point at the note | "A list. That is why there are square brackets inside the round ones. And that is why every layer ends with a comma." |
| Click "Compile" | "Move two: compile. Three choices. How to update the weights. What to measure as error. What to report. Nothing is learned yet. Compile only sets the rules." |
| Click "Fit" | "Move three: fit. This is the only move that learns. Training pictures, their labels, how many epochs, and a validation set to check on." |
| Point at the warning | "And one warning. Fit is the only move that is not safe to run twice. A second fit does not start again. It continues." |
| Click "Evaluate" | "Move four: evaluate. Pictures the model has never seen. It returns two numbers: the loss, then the accuracy. In that order." |
| Click "Predict" | "Move five: predict. Pictures only, no labels. It returns probabilities: one row per picture, one number per class. Then argmax picks the largest." |
| Point along the five boxes | "Build, compile, fit, evaluate, predict. Every Keras program you will read has this shape. Find the five moves first, and the code stops looking like a wall." |

## Episode 8 Part 2: One layer, read slowly (1:39–2:13)

| Show | Say |
| --- | --- |
| Scroll to "One layer, read slowly". Click "Conv2D" | "Now one layer, read slowly. Conv2D. Capital C, capital D. Python is case-sensitive. A small c is a different name." |
| Click "(3, 3)" | "The filter size. On paper we write 3 times 3. In Python it is two numbers with a comma between them, inside their own brackets." |
| Click activation="relu" | "A named setting. The value is in quotes, and it is lowercase. Keras looks up the word exactly as you wrote it." |
| Click the last comma | "And the last comma. The easiest one to forget. This layer is one item in a list. Items in a list are separated by commas." |

## Episode 8 Part 3: Nine bugs (2:13–3:47)

| Show | Say |
| --- | --- |
| Scroll to "Spot the bug". Bug 1: click the first line | "Now the bugs. Number one. A small s in Sequential. Python says the name is not defined. It is not being difficult. To Python, that is a different word." |
| Bug 2: click the "loss" line (the wrong guess) | "Number two. Python points at this line, the loss line. But this line is fine." |
| Click the "optimizer" line | "The comma is missing on the line above. Python only notices one line later. So always check the line above the arrow." |
| Bug 3: click the Conv2D line | "Number three. I copied the instructions: 3, multiplication sign, 3. Python cannot read that symbol. A size is two numbers and a comma." |
| Bug 5: click the last Dense line | "Number five. Softmax, in capital letters. Every bracket is correct, and it still fails. Names inside quotes must be lowercase." |
| Bug 6: click the last line | "Number six. A bracket in the wrong place. Float takes one thing. Round takes two. The message says exactly that: float got two arguments. Read brackets from the inside out." |
| Bug 7: click the "deep" line | "Number seven. Three pairs, but the third bracket closes too early. So Python tries to split the word deep into two parts. Too many values to unpack." |
| Bug 8: click the compile line | "Number eight. I copied a cell and changed the model's name. In most places. Not all. The old model was compiled twice. The new one, never." |
| Bug 9: do not click yet | "And number nine. The dangerous one. I ran the training cell twice. No error. No warning. Can you see the proof?" |
| Click the first line of the second run | "Epoch one, accuracy 65 percent. A new model cannot do that. It was not epoch one. It was epoch six. The model just kept training. These are real numbers from my notebook." |

## Episode 8 Part 4: Three habits (3:47–4:08)

| Show | Say |
| --- | --- |
| Scroll to "Three habits". Point at the first | "Three habits prevent most of this. One: copy, do not retype. Eight of my nine bugs came from typing by hand." |
| Point at the second | "Two: read the last line of the error first. That is where the useful sentence is." |
| Point at the third | "Three: before you trust a result, restart and run all. It is the only way to know that the code on the page made the numbers on the page." |

## Episode 8 wrap-up (4:08–4:18)

| Show | Say |
| --- | --- |
| Back to the first panel | "Five moves. Nine bugs. Three habits. The page is in the description. Try to find all nine without the Show me button. See you in the next one." |

## Episode 8 recording checklist

- [ ] Check the live link opens, and reload the page so "Found 0 of 9" shows
- [ ] Browser zoom at 125%, light mode, bookmarks bar hidden, notifications off
- [ ] Bug 2: click the "loss" line first on purpose, to show the hint, then the "optimizer" line
- [ ] Bug 4 is used in the hook and skipped in Part 3
- [ ] Publishing on BrokenLabsFix (Hindi/Bahasa audience, auto-dubbing on)

## Episode 8 title options

1. Keras in 5 Moves, and 9 Bugs Every Beginner Makes | Deep Learning Without the Formula Wall Ep. 8
2. 9 Keras Errors Explained: Commas, Brackets and Capital Letters | Deep Learning Without the Formula Wall Ep. 8
3. Build, Compile, Fit, Evaluate, Predict: The Keras Pattern | Deep Learning Without the Formula Wall Ep. 8

## Episode 8 description

```
One free interactive page on writing Keras code. The five moves every Keras program follows, one layer read piece by piece, and nine real beginner bugs with the exact Python error message for each. Click the line you think is wrong. No install.

Keras in five moves, and nine ways to break it
https://fleabyte26.github.io/Introduction-to-Deep-Learning/keras-patterns.html

Companion page, the results of the three models:
https://fleabyte26.github.io/Introduction-to-Deep-Learning/cnn-three-models.html

Episode 7 (three CNNs, five epochs): https://www.youtube.com/watch?v=02nDUhqnNik

All pages and source: https://github.com/Fleabyte26/Introduction-to-Deep-Learning

0:00 One wrong character
0:27 The five moves: build, compile, fit, evaluate, predict
1:39 One layer, read slowly
2:13 Nine real bugs and their error messages
3:47 Three habits
4:08 Wrap-up
```

## Episode 8 tags

keras tutorial, keras for beginners, python errors, SyntaxError, NameError, TypeError, ValueError, AttributeError, Sequential model, model.fit, model.compile, jupyter notebook, debugging python, tensorflow, CNN, deep learning, machine learning tutorial, interactive explainer

# Episode 9: Recurrent Neural Networks

BrokenLabsFix · Kris Hornung · November 2026

Watch the recorded video: https://www.youtube.com/watch?v=LT6A6pNxh68

## Episode 9 at a glance

A 4-minute walkthrough of one page. A tiny character-level recurrent network, trained on two sentences, runs live in the page. The episode steps through a sentence one letter at a time, shows where the network stores what it must remember, and then shows why that memory fades with distance.

First of three episodes on sequences: recurrent networks, then LSTM, then attention. Made the same way as Episodes 6 to 8: the page first, then this script, then the video rendered from the page with a synthetic narration voice.

| Part | Time | What happens |
| --- | --- | --- |
| Hook | 0:00–0:36 | It remembers the animal |
| Part 1 | 0:36–1:47 | One letter, one update |
| Part 2 | 1:47–2:03 | Where the animal is stored |
| Part 3 | 2:03–2:28 | More steps, not more weights |
| Part 4 | 2:28–3:02 | Scoring a guess |
| Part 5 | 3:02–3:53 | Why memory fades |
| Wrap-up | 3:53–4:09 | Three things to keep |

Page: [rnn.html](https://fleabyte26.github.io/Introduction-to-Deep-Learning/rnn.html)

Same format: left column is what to do on screen, right column is what to say. Every number is printed on the page.

## Episode 9 hook (0:00–0:36)

| Show | Say |
| --- | --- |
| The page, scrolled to "The memory test" | "Two sentences. The cat sat on the mat. The dog sat on the log. A tiny neural network reads them, one letter at a time." |
| Point at the three boxes | "After the words, sat on the, it has to choose. M for mat, or L for log. It gets it right every time. One hundred percent." |
| Same | "But those last eleven letters are identical in both sentences. The only clue is the animal, and that was fifteen letters ago. So the network must be remembering. Today: how a network remembers." |
| Scroll to the top of the page | "This is a recurrent neural network. It is chapter 9 of Dive into Deep Learning. And the network on this page is real. It runs in your browser." |

## Episode 9 Part 1: Step through a sentence (0:36–1:47)

| Show | Say |
| --- | --- |
| The first panel. Press Reset | "Here is its memory. Twenty-four numbers. Right now they are all zero. It has read nothing." |
| Press Step once | "Press Step. It reads the letter T. It mixes that letter with its old memory, and gets a new memory. Then it guesses the next letter: H." |
| Press Step twice more | "Read H. Read E. Each time, the same small calculation. New memory equals old memory, plus one new letter." |
| Press Step. It has read "the" and a space | "Now a space. The next word is cat, or dog. And look at its guess. C and D, about half each. It does not know. And that is correct. Nobody could know." |
| Press Step. It reads "c". Point at the two memory rows | "It reads C. Now it knows the animal. Look at the two memory rows. Before, and after. The memory changed. The cat is now stored in those numbers." |
| Press Play, or Step fourteen more times, to the end of "the " | "Now watch it read the rest. A, T. Sat. On. The. Fourteen more letters. The memory changes at every step, but the cat has to survive inside it." |
| Point at the guess | "And here is the moment. Next letter: M. One hundred percent. It remembered." |
| Click the dog sentence and press Play | "Same thing with the dog sentence. The same last eleven letters. This time, it says L. For log." |
| Press "Let it write" | "And it can write by itself. Each letter it chooses goes back in as the next input. The orange letters are its own." |

## Episode 9 Part 2: The memory test (1:47–2:03)

| Show | Say |
| --- | --- |
| Scroll to "The memory test". Point at the two rows | "So where is the animal stored? Here are the two memories, at the moment of decision. Eight of the twenty-four numbers are clearly different. They are outlined in black." |
| Point at the verdict box | "Nobody told the network to store the animal. It learned to, because that was the only way to get the last word right." |

## Episode 9 Part 3: One set of weights (2:03–2:28)

| Show | Say |
| --- | --- |
| Scroll to "One set of weights, used at every step" | "Drawn along a sentence, the network looks like a long chain. It is not. It is one small network, used again at every letter." |
| Drag "Sentence length" from 24 to 200 | "This network has 1,286 weights. For a sentence of 24 letters: 1,286. For 200 letters: still 1,286. A longer sentence means more steps, not more weights." |

## Episode 9 Part 4: Perplexity (2:28–3:02)

| Show | Say |
| --- | --- |
| Scroll to "How good is a guess: perplexity" | "How do we score it? With perplexity. On average, how many letters is it still choosing between? Pure guessing among 14 letters gives 14. Always right gives 1." |
| Point at the three boxes | "Before training: 16. After training: 1.03." |
| Point at the row of bars, then the one short bar | "Why not exactly 1? Look at the bars. Every one is full, except one. The first letter of the animal. That one is a coin flip. And no network can predict a coin flip." |
| Point at the verdict box | "The best possible score here is 1.029. The network scores 1.030. It learned everything there was to learn." |

## Episode 9 Part 5: Backpropagation through time (3:02–3:53)

| Show | Say |
| --- | --- |
| Scroll to "Backpropagation through time" | "Last question. How did it learn that M depends on cat? The error at the M must travel backwards, through every step in between. This is called backpropagation through time." |
| Slider at 0.85. Point at the numbers | "And each step multiplies the signal. Multiply by 0.85, fifteen times, and only 9 percent is left." |
| Point at the black line | "The black line is measured from the real network. After fifteen steps, 26 percent of the signal survives. After twenty-four steps, less than 3 percent. That was enough for cat and mat. For a clue one hundred steps back, nothing would arrive. This is the vanishing gradient." |
| Drag the slider above 1.00 | "And if the multiplier is above 1, the opposite happens. The signal explodes. That is why recurrent networks use gradient clipping." |
| Set the slider to exactly 1.00 | "Exactly 1 is the only safe value. And holding it at exactly 1, on purpose, is the idea behind the LSTM. That is the next episode." |

## Episode 9 wrap-up (3:53–4:09)

| Show | Say |
| --- | --- |
| Back to the first panel | "So. A recurrent network reads one piece at a time, and carries a memory forward. It uses the same weights at every step. And its memory fades with distance." |
| Same | "The page is in the description. Step through both sentences, and let it write. See you in the next one." |

## Episode 9 recording checklist

- [ ] Check the live link opens
- [ ] Browser zoom at 125%, light mode, bookmarks bar hidden, notifications off
- [ ] Press Reset before stepping, and use Step (not Play) for the first five letters so you control the pace
- [ ] Publishing on BrokenLabsFix (Hindi/Bahasa audience, auto-dubbing on)

## Episode 9 title options

1. Recurrent Neural Networks Explained: A Network With a Memory | Deep Learning Without the Formula Wall Ep. 9
2. How an RNN Remembers, and Why It Forgets | Deep Learning Without the Formula Wall Ep. 9
3. RNN, Perplexity and Vanishing Gradients, Step by Step | Deep Learning Without the Formula Wall Ep. 9

## Episode 9 description

```
One free interactive page with a real recurrent neural network running in your browser. Step through a sentence one letter at a time, watch its 24-number memory change, see where it stores what it needs to remember, and see why that memory fades. No install, no code.

Recurrent neural networks: a network with a memory
https://fleabyte26.github.io/Introduction-to-Deep-Learning/rnn.html

All pages and source: https://github.com/Fleabyte26/Introduction-to-Deep-Learning

Based on Dive into Deep Learning (d2l.ai), sections 9.4 and 9.7. Next episode, LSTM: https://www.youtube.com/watch?v=rgTcmLxvKNY

0:00 It remembers the animal
0:36 Step through a sentence
1:47 Where the memory is stored
2:03 One set of weights at every step
2:28 Perplexity: scoring a guess
3:02 Backpropagation through time and the vanishing gradient
3:53 Wrap-up
```

## Episode 9 tags

recurrent neural network, RNN, RNN explained, hidden state, backpropagation through time, BPTT, vanishing gradient, exploding gradient, gradient clipping, perplexity, language model, character-level RNN, LSTM, sequence models, deep learning, d2l, dive into deep learning, machine learning tutorial, interactive explainer

# Episode 10: LSTM: A Memory With Gates

BrokenLabsFix · Kris Hornung · November 2026

Watch the recorded video: https://www.youtube.com/watch?v=rgTcmLxvKNY

## Episode 10 at a glance

A 4-minute walkthrough of one page. A plain recurrent network and an LSTM, trained on the same sentences, are compared as the sentence gets longer. Then one memory cell with its three gates operated by hand, and the real trained LSTM stepped through a sentence while one cell holds the fact it has to remember.

Second of three episodes on sequences: recurrent networks, LSTM, attention.

Made the same way as Episodes 6 to 9: the page first, then this script, then the video rendered from the page with a synthetic narration voice.

| Part | Time | What happens |
| --- | --- | --- |
| Hook | 0:00–0:29 | The plain network forgets |
| Part 1 | 0:29–0:46 | Longer than anything in training |
| Part 2 | 0:46–1:40 | Keep, write, show |
| Part 3 | 1:40–2:43 | One cell holds the animal |
| Part 4 | 2:43–3:14 | A multiplier held near one |
| Part 5 | 3:14–3:25 | Four times the weights |
| Wrap-up | 3:25–3:45 | And what comes next |

Page: [lstm.html](https://fleabyte26.github.io/Introduction-to-Deep-Learning/lstm.html)

Same format: left column is what to do on screen, right column is what to say. Every number is printed on the page.

## Episode 10 hook (0:00–0:29)

| Show | Say |
| --- | --- |
| The page, first panel, slider at 0 | "Last episode, a small network remembered an animal across fifteen letters. The cat sat on the mat." |
| Drag "Extra words" slowly to 5 | "Now make the sentence longer. The cat, very really quite very really, sat on the mat. The same network design. The same task." |
| Point at the three boxes | "The plain network is right 45 percent of the time. That is a coin flip. It has forgotten the animal. That is the orange line." |
| Point at the blue line | "The blue line is an LSTM. One hundred percent. Today: what an LSTM does differently." |

## Episode 10 Part 1: The comparison (0:29–0:46)

| Show | Say |
| --- | --- |
| Drag the slider on to 14 | "And look at the right side. Neither network ever saw a sentence with more than eight extra words. The LSTM is still right at fourteen. It did not memorise lengths. It learned how to hold on to something." |
| Point at "trained for 80 passes" and "trained for 12 passes" | "And this was not a fair fight. The plain network got nearly seven times more training." |

## Episode 10 Part 2: One cell, three gates (0:46–1:40)

| Show | Say |
| --- | --- |
| Scroll to "One memory cell, three gates". Click "Keep" | "So what is inside? A memory cell with three gates. A gate is just a number between zero and one. It multiplies something. Zero blocks it. One lets it through." |
| Point at the numbers panel | "Here is the whole rule. New memory equals forget gate times old memory, plus input gate times new information." |
| Point at the verdict box | "Preset one: keep. Forget gate at one. Input gate at zero. The old memory passes straight through, unchanged. This is the move a plain network does not have." |
| Click "Overwrite" | "Overwrite. Forget gate zero, input gate one. The old memory is thrown away, and the new information takes its place." |
| Click "Add to it" | "Add. Both gates open. The new information is added to the old." |
| Click "Keep, but hide" | "And the third gate. Keep, but hide. The output gate is zero. The memory is still there, 0.8, but the rest of the network sees nothing. The cell can carry a fact silently, and show it only when it is needed." |

## Episode 10 Part 3: Inside the trained LSTM (1:40–2:43)

| Show | Say |
| --- | --- |
| Scroll to "Inside the trained LSTM". "5 extra words", Reset | "Now the real LSTM. It has 24 cells. Training picked one of them, cell 8, to hold the animal. Let's watch that one cell. Blue is the cat sentence. Orange is the dog sentence." |
| Press Step four times | "The. Space. Both sentences are identical so far, so the two lines are on top of each other." |
| Press Step four more times | "Now the animal. C-A-T, or D-O-G. And the lines split apart. The cell is being written." |
| Press Play and pause at the end of the extra words | "Now the extra words. Thirty letters. The lines wobble, because this cell does other small jobs too. But they never cross. And look at the forget gate. Almost one, the whole way." |
| Point at the output gate | "Watch the output gate as well. It opens and closes. The cell holds the animal the whole time, but only shows it at certain letters." |
| Click "Jump to the end" | "And at the end, after 45 letters: plus seven for cat, minus ten for dog. The network reads the answer straight from that cell. M for mat. L for log." |
| Click "12 extra words", then "Jump to the end" | "Twelve extra words. 87 letters. Longer than anything it was trained on. The lines are still apart." |

## Episode 10 Part 4: Why it does not fade (2:43–3:14)

| Show | Say |
| --- | --- |
| Scroll to "Why it does not fade". Point at the orange line | "Why does it not fade? Last episode, the training signal in the plain network was multiplied at every step. After 24 steps, less than 3 percent was left. That is the orange line." |
| Point at the numbers panel | "In the LSTM cell, the path back through the memory is multiplied only by the forget gate. And that gate averages 0.994. After sixty steps, 70 percent of the signal is still there." |
| Point at the verdict box | "That is the whole trick. A multiplier that training can hold close to one, for exactly the cells that need to remember." |

## Episode 10 Part 5: What it costs (3:14–3:25)

| Show | Say |
| --- | --- |
| Scroll to "What it costs" | "It is not free. Three gates, plus the new information: four small networks where there was one. Four times the weights in that layer, and four times the work at every step." |

## Episode 10 wrap-up (3:25–3:45)

| Show | Say |
| --- | --- |
| Back to the gates panel, "Keep" | "So. An LSTM is a memory with three gates. Forget, input, output. Keep, write, show." |
| Same | "But even an LSTM squeezes the whole sentence into one small memory. The next idea is different. Do not squeeze at all. Just look back. That is attention, and it is the next episode." |

## Episode 10 recording checklist

- [ ] Check the live link opens
- [ ] Browser zoom at 125%, light mode, bookmarks bar hidden, notifications off
- [ ] Publishing on BrokenLabsFix (Hindi/Bahasa audience, auto-dubbing on)

## Episode 10 title options

1. LSTM Explained: A Memory With Three Gates | Deep Learning Without the Formula Wall Ep. 10
2. Why LSTMs Remember and Plain RNNs Forget | Deep Learning Without the Formula Wall Ep. 10
3. Forget Gate, Input Gate, Output Gate: LSTM Step by Step | Deep Learning Without the Formula Wall Ep. 10

## Episode 10 description

```
One free interactive page with a real LSTM running in your browser. Compare it with a plain recurrent network as the sentence gets longer, work one memory cell's three gates by hand, and watch the trained network hold a fact in a single cell for 87 letters. No install, no code.

LSTM: a memory with gates
https://fleabyte26.github.io/Introduction-to-Deep-Learning/lstm.html

All pages and source: https://github.com/Fleabyte26/Introduction-to-Deep-Learning

Based on Dive into Deep Learning (d2l.ai), section 10.1. Previous episode, recurrent neural networks: https://www.youtube.com/watch?v=LT6A6pNxh68
Next episode: attention.

0:00 The plain network forgets
0:29 Longer than anything in training
0:46 One cell, three gates
1:40 Inside the trained LSTM
2:43 Why it does not fade
3:14 What it costs
3:25 Wrap-up
```

## Episode 10 tags

LSTM, LSTM explained, long short-term memory, forget gate, input gate, output gate, memory cell, cell state, vanishing gradient, recurrent neural network, RNN vs LSTM, sequence models, gated recurrent network, deep learning, d2l, dive into deep learning, machine learning tutorial, interactive explainer

# Episode 11: Attention: Do Not Remember, Look Back

BrokenLabsFix · Kris Hornung · November 2026

## Episode 11 at a glance

A 4-minute walkthrough of one page. A small two-head attention network, trained on the same sentences as the recurrent network and the LSTM, runs live in the page. The episode shows where each head looks, works a query against keys and values by hand, and then puts all three networks on one test.

Third of three episodes on sequences: recurrent networks, LSTM, attention.

Made the same way as Episodes 6 to 9: the page first, then this script, then the video rendered from the page with a synthetic narration voice.

| Part | Time | What happens |
| --- | --- | --- |
| Hook | 0:00–0:39 | It just looks |
| Part 1 | 0:39–1:27 | Two heads, two jobs |
| Part 2 | 1:27–2:20 | A soft lookup |
| Part 3 | 2:20–3:05 | Distance, and cost |
| Wrap-up | 3:05–3:30 | From attention to the transformer |

Page: [attention.html](https://fleabyte26.github.io/Introduction-to-Deep-Learning/attention.html)

Same format: left column is what to do on screen, right column is what to say. Every number is printed on the page.

## Episode 11 hook (0:00–0:39)

| Show | Say |
| --- | --- |
| The page, first panel, cat sentence, 5 extra words | "The same sentence as before. The cat, very really quite very really, sat on the. What comes next?" |
| Point at the orange row, and the one dark square | "This network does not carry the animal through the extra words. It looks back. The orange row shows where. 93 percent of its attention is on one letter, 42 letters back. The last letter of cat." |
| Point at the guess | "And it answers M, with 99.7 percent. No memory cell. No gates. It just looked. This is attention." |
| Scroll to the top of the page | "This is the third episode of three. A plain network forgets. An LSTM protects its memory. Attention does not remember at all. This is chapter 11 of Dive into Deep Learning." |

## Episode 11 Part 1: Where it looks (0:39–1:27)

| Show | Say |
| --- | --- |
| Click the dog sentence | "Switch to the dog sentence. The same head looks at the same place, finds a G instead of a T, and the answer changes to L." |
| Back to cat. Drag "Extra words" to 12 | "Make the sentence longer. Twelve extra words. Now the animal is 84 letters back. For attention, that is still one step." |
| Point at the blue row | "There are two heads. Head one, in blue, looks somewhere else: at the N of the word, on. That tells it this is the second, the, in the sentence, so a noun is coming." |
| Point at both rows | "Two heads, two jobs. One finds what kind of word is due. The other finds which animal. Nobody assigned those jobs. Training did. This is multi-head attention." |
| Set extra words back to 5. Click a letter in the middle of the sentence | "And every letter does its own looking. Click any letter. From the middle of a word, head one looks at the letters just before it. Head two is still looking at the animal. It keeps the animal in view from almost anywhere." |

## Episode 11 Part 2: Queries, keys and values (1:27–2:20)

| Show | Say |
| --- | --- |
| Scroll to "Queries, keys and values, by hand" | "How does it decide where to look? With three things. Keys, values, and a query. Every stored item has a key: what it is about. And a value: what it holds." |
| Point at the four bars | "The query is the question. It is compared with every key. Here the query points near the animal key, so the animal gets the biggest share: 94 percent." |
| Drag "Query direction" round to the place key | "Turn the query. Now it points at the place key. The shares move with it. The answer becomes, mat." |
| Point at the numbers panel | "The shares always add up to one hundred percent. That is a softmax. And the answer is the values, mixed in those shares." |
| Drag "Sharpness" all the way down | "Now lower the sharpness. Every key gets about the same share. The answer is a blur of everything. The question is being ignored." |
| Drag "Sharpness" all the way up | "Raise it. One key takes everything. Now it is an ordinary lookup. Real attention lives in between. And because every step is smooth, the network can learn where to look." |

## Episode 11 Part 3: Three memories, one test (2:20–3:05)

| Show | Say |
| --- | --- |
| Scroll to "Three memories, one test" | "Now all three networks, on the same test. Orange, the plain network: a coin flip after one extra word. Blue, the LSTM: perfect. Green, attention: perfect too, up to eight extra words." |
| Point at the right side of the chart, then the verdict | "But look at the right side. Past the lengths it trained on, attention slips. The LSTM does not. Attention removed the distance problem. It did not remove the need to train on what you will be tested on." |
| Point at the middle row of the table | "Here is the real difference. In a recurrent network, the animal is 42 steps from the answer. With attention, it is one step. At any distance." |
| Drag "Sentence length" to 1,000 | "And the cost. For a thousand letters, a recurrent network takes a thousand steps. Attention makes half a million comparisons. But it can do them all at the same time." |

## Episode 11 wrap-up (3:05–3:30)

| Show | Say |
| --- | --- |
| Scroll to "The rest of the chapter" | "Attention started in translation, as an add-on to recurrent networks. Then people asked: what if we remove the recurrent network, and keep only attention? That is the transformer. It is behind every large language model today." |
| Back to the first panel | "So. Three ways to remember. Rewrite it at every step, and lose it. Lock it in a cell. Or do not carry it at all, and look back. The page is in the description. See you in the next one." |

## Episode 11 recording checklist

- [ ] Check the live link opens
- [ ] Browser zoom at 125%, light mode, bookmarks bar hidden, notifications off
- [ ] Publishing on BrokenLabsFix (Hindi/Bahasa audience, auto-dubbing on)

## Episode 11 title options

1. Attention Explained: Queries, Keys and Values | Deep Learning Without the Formula Wall Ep. 11
2. Attention vs LSTM vs RNN: Three Ways to Remember | Deep Learning Without the Formula Wall Ep. 11
3. How Attention Works, With a Real Network You Can Click | Deep Learning Without the Formula Wall Ep. 11

## Episode 11 description

```
One free interactive page with a real attention network running in your browser. Click any letter to see where its two heads look, turn a query against four keys by hand, and compare attention with an LSTM and a plain RNN on the same test. No install, no code.

Attention: do not remember, look back
https://fleabyte26.github.io/Introduction-to-Deep-Learning/attention.html

All pages and source: https://github.com/Fleabyte26/Introduction-to-Deep-Learning

Based on Dive into Deep Learning (d2l.ai), sections 11.1 to 11.5. Previous episodes: recurrent neural networks https://www.youtube.com/watch?v=LT6A6pNxh68 and LSTM https://www.youtube.com/watch?v=rgTcmLxvKNY

0:00 It just looks
0:39 Where it looks: two heads, two jobs
1:27 Queries, keys and values
2:20 Three memories, one test
3:05 From attention to the transformer
```

## Episode 11 tags

attention mechanism, attention explained, queries keys values, self-attention, multi-head attention, scaled dot-product attention, softmax, Bahdanau attention, transformer, RNN vs LSTM vs attention, sequence models, large language models, deep learning, d2l, dive into deep learning, machine learning tutorial, interactive explainer
