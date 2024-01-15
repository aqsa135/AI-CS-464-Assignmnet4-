### CS 462 - Assignment 4. Learning and Representing Knowledge.

#### Due Monday Oct 23, 1:00pm 

_I have added optional milestone recommendations for each assignment to help you organize your time. These are just suggestions._


To submit: Please check all code into your GitHub repo, along with a PDF containing the 
answers to the written questions. Also, please provide a submission.py that demonstrates
all of your code.

_(Milestone 1: Mon Oct 9. Check out code, read assignment)_

#### Part 1. Inference. 

NASA has reached out to us for help in designing the logic for the Mars rover.
They would like it to be able to conduct experiments, and only radio back to Earth for help 
in case of an actual emergency. We've decided to build a rule-based system to implement this.

We have the following predicates (with abbreviations) to use:

- a - battery failed
- b - solar panels failed
- c - base station failed
- d - backup battery failed
- e: send emergency signal
- q: backup switch fails
- w: cannot sync base station with rover
- x: battery sensor not responding
- y: battery will not power motor
- z: no current from panels

NASA has given us the following rules to use:
- If the battery sensor is not responding and the battery will not power the motor, 
then the battery failed.
- If the battery sensor is not responding, we are not able to sync the base station with the rover.
- If the battery sensor is not responding and there is no current from the panels, then the solar panels have failed.
- If the battery will not power the motor, and we cannot sync the base station with the rover, then the base station has failed.
- If the backup switch fails, and there's no current from the panels, and the battery fails, then the backup battery fails.
- If the backup switch fails, then the battery will not power the motor.
- If battery failed and solar panels failed and base station failed and backup battery failed, then send the emergency signal.
- If we cannot sync the base station with the rover, we are unable to get current from the panels.

Our rover has observed the following facts:
- The battery sensor is not responding.
- The backup switch has failed.

**(5 points)** *(milestone 2 Wed Oct 11)* 1. Show what the initial KB looks like, using the abbreviations for each rule and fact.



**(5 points)** *(milestone 2 Wed Oct 11)* 2. Use forward chaining to show that we should send an emergency signal. For each step, show the rule and fact(s) being matched, and the resulting fact.
for example: q, q->y, y.



**(5 points)** *(milestone 2 Wed Oct 11)*  3. Our NASA contact is skeptical; he's heard that AI can make mistakes, and does not believe that there is truly an emergency.
We need to prove that this was the right decision. Use backward chaining to show that signaling the emergency was the correct decision. For each
step, show the stack and the items to be proved.


4. Our NASA contact is still skeptical - just because we were right this time doesn't mean it could never make mistakes, he argues!
We need to prove that our rules will always generate the right result. We'll use resolution to verify the correctness of our rules.

**(5 points)** *(milestone 3 Fri Oct 13)* a) Convert the KB to Conjunctive Normal Form.



**(5 points)** *(milestone 3 Fri Oct 13)* b) Add the term !e (do not signal) to the KB, and use resolution to derive a contradiction.
For each step, show the two terms being resolved and the new sentence that results. 
for example: (!w v z) ^ (!x v w) -> (z v !x)


#### Part 2: Decision tree 

This is the largest part of the assignment - we'll implement the basic decision tree algorithm. I've provided two starter datasets (tennis and restaurant) for you, plus some unit tests.
Start with these datasets, and then test on breast cancer once you've got it working.

*(milestone 2 Wed Oct 11 - look at the code, start to figure it out)*

**(5 points)** *(milestone 3 Fri Oct 13)* I've implemented entropy for you. Use that to implement remainder.



**(10 points)** *(milestone 4 Sat Oct 14)* Use remainder to implement select_attribute.



**(10 points)** *(milestone 5 Sun Oct 15)* Use select_attribute to implement make_tree.



**(10 points)** *(milestone 6 Mon Oct 16)* Now you are ready to implement classify. 


**(10 points)** *(milestone 7 Wed Oct 18)* Next, measure the performance of your tree on both the restaurant and the tennis data using five-fold cross-validation. You can use either your own implementation or one of the ones from a previous assignment.

**(5 points)** *(milestone 8 Fri Oct 20 here down)* Once that's working, try your tree out on the [California Housing Dataset](https://scikit-learn.org/stable/datasets/real_world.html) from sklearn. Again, measure the performance using five-fold cross-validation.

**(5 points)** Now, compare your tree to the [sklearn DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html). Use 'entropy' as the splitting criterion to get the fairest performance.

**(5 points)** Last, let's see how adding multiple classifiers to create a Random Forest changes things. Compare your tree and the sklearn tree to the [sklearn RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier). Again, use entropy as the criterion in order to get a 
fair comparison.



#### Part 3: Knowledge Graphs 

*(milestone 8 Fri Oct 20 - we will do this as part of a case study on Fri Oct 20)*

Wikipedia sits on top of a knowledge graph (called WikiData). In this question, you'll get a chance to explore this knowledge graph.

To begin, open the page for [the Perseverance rover](https://en.wikipedia.org/wiki/Perseverance_(rover)). Go to 'tools' (on the right) and then WikiData to bring up the data page.

Every WikiData object has a unique id - this one is Q87749354. 

Scroll down to see some of the statements associated with this object. 
Note that each of the relations is also an object. For example, click on 'video' to learn what this link means.

We can search this graph in one of two ways: using a query language called SPARQL, or with a web-based tool. 
We'll do the latter, but if you're familiar with SQL, check out SPARQL.

**(5 points)** To start, go to the [Wikipedia Query Builder](https://query.wikidata.org/querybuilder/?uselang=en) and enter "instance of" and "Mars Rover" in the Property and Value boxes.
How many results come up? What are they?

Now add a second condition, that it must be a part of the Mars 2020 mission. Now how many results do you get?

**(10 points)** Now let's visualize this. Click on the 'graph' icon (it looks like arrows in a loop) and then select the Perseverance node. Notice that you can expand the nodes that are connected to Perseverance to answer questions.
Use this to determine the answers to the following questions:
- What type of space launch vehicle carried Perseverance to Mars, and what country built it?
- Who was the launch contractor for the Mars 2020 mission? What companies own them?
- what mission does Perseverance follow? Where did it land?
- What powers Perseverance? What two organizations designed it?



#### Part 4: (686 students only) 

Please read [this article](https://12ft.io/proxy?q=https%3A%2F%2Fwww.theatlantic.com%2Fmagazine%2Farchive%2F2013%2F11%2Fthe-man-who-would-teach-machines-to-think%2F309529%2F) about Douglas Hofstadter, 
which also serves as a nice summary of the history of AI and the debates 
over the value of developing machines that think like humans. 
(As an aside: If you have not read Hofstadter's book [Godel, Escher, Bach](https://en.wikipedia.org/wiki/G%C3%B6del,_Escher,_Bach), 
I strongly recommend it.)

Prepare a summary or critique of this article that addresses the 
following questions:
- Hofstadter is particularly interested in understanding the way humans think. What sorts of reasoning mechanisms does he study?
- The article includes a quote from our text: “The quest for ‘artificial flight’ succeeded when 
the Wright brothers and others stopped imitating birds and started … learning about aerodynamics,” What does this mean? Why is it relevant to AI?
- What was Candide? Why did it change the way we thought about machine translation?
- The article also contains a quote from the last chapter of AIMA: perhaps AI has become too much like 
the man who tries to get to the moon by climbing a tree: “One can report steady progress, all the way to the top of the tree.” What does this mean? How does it relate to Candide 
and the ways in which big data and machine learning have changed AI?
