# w261 Machine Learning at Scale
This is the base repo for all course materials for **UC Berkeley School of Information, MIDS w261**. Please consider this document ground truth for reading assignments & deadlines.

## Quick Links
* [Syllabus](https://docs.google.com/document/d/1BTbc6znZe3wTpdIVeun66K8AkzUlfYW1dlDPZkbZ5NM/edit#heading=h.jqn77w17bzrl): _course overview & grading policies_
* [Class Schedule](#class-schedule): _dates, instructors & deadlines_
* [Weekly Materials & Assignments](#weekly-materials-and-assignments): _readings, videos, notebooks_
* [Environment Repo](https://github.com/UCB-w261/w261-environment): _OPTIONAL Docker and GCP setup help_    
* [HW Survey](https://docs.google.com/forms/d/e/1FAIpQLSce9feiQeSkdP43A0ZYui1tMGIBfLfzb0rmgToQeZD9bXXX8Q/viewform) _optional survey for each HW assignment. Suggestions, concerns, difficulties, etc._

## Class Schedule

__Office Hours:__  
Friday 6pm - 7pm PST (w/Ramki)<br>
Thursday 12pm - 1pm PST (w/Kyle)<br>
TBD (w/Jimi)<br>
TBD (w/Luis)<br>
TBD (w/Padma)<br>

<table border="1" class="dataframe">
<thead>
<tr>
<th>Date</th>
<th>Week</th>
<th>Topic</th>
<th>Due</th>
</tr>
</thead>
<tbody>
<tr>
<td>Jan 8, 9</td>
<td>1</td>
<td>Intro to Machine Learning at Scale</td>
<td></td>
</tr>
<tr>
<td>Jan 15, 16</td>
<td>2</td>
<td>Parallel Computation Frameworks</td>
<td>HW 1 due 24hrs before your live session</td>
</tr>
<tr>
<td>Jan 22, 23</td>
<td>3</td>
<td>MapReduce Algorithm Design</td>
<td></td>
</tr>
<tr>
<td>Jan 29, 30</td>
<td>4</td>
<td>Intro to Spark/MapReduce with RDDs </td>
<td>HW 2 due 24hrs before your live session</td>
</tr>
<tr>
<td>Feb 5, 6</td>
<td>5</td>
<td>Spark/MapReduce with RDDs (con't)</td>
<td></td>
</tr>
<tr>
<td>Feb 12, 13</td>
<td>6</td>
<td>Distributed Supervised ML (part 1)</td>
<td>HW 3 due 24hrs before your live session</td>
</tr>
<tr>
<td>Feb 19, 20</td>
<td>7</td>
<td>Distributed Supervised ML (part 2)</td>
<td>(Project team assignments)</td>
</tr>
<tr>
<td>Feb 26, 27</td>
<td>8</td>
<td>Spark Optimizations for Big Data and DataFrames</td>
<td>HW 4 due 24hrs before your live session</td>
</tr>
<tr>
<td>Mar 4, 5</td>
<td>9</td>
<td>Graph Algorithms at Scale (part 1)</td>
<td></td>
</tr>
<tr>
<td>Mar 11, 12</td>
<td>10</td>
<td>Graph Algorithms at Scale (part 2)</td>
<td></td>
</tr>
<tr>
<td>Mar 18, 19</td>
<td>11</td>
<td>ALS and Spark MLLib</td>
<td>HW 5 due 24hrs before your live session</td>
</tr>
<tr>
<td>Mar 25, 26</td>
<td>NO CLASSES</td>
<td></td>
<td></td>
</tr>
<tr>
<td>Apr 1, 2</td>
<td>12</td>
<td>Decision Trees</td>
<td></td>
</tr>
<tr>
<td>Apr 8, 9</td>
<td>13</td>
<td>Spark Online Learning</td>
<td></td>
</tr>
<tr>
<td>Apr 13, 14</td>
<td>14</td>
<td>Final Project Presentations</td>
<td>Final Project due 24hrs before your live session</td>
</tr>

</tbody>
</table>


## Weekly Materials And Assignments:
_Reading Assignment Abbreviations:_   
> "**HDG**" = <u>Hadoop: Definitive Guide (4th Edition)</u> by Tom White    
"**SDG**" = <u>Spark: The Definitive Guide: Big Data Processing Made Simple</u> By Bill Chambers, Matei Zaharia    
"**DITP**"" = <u>Data Intensive Text Processing With Map Reduce</u> by Lin & Dyer      
"**IIR**" = <u>Introduction to Information Retrieval</u> by Manning, Raghavan, & Shutze   
"**ISL**" = <u>Introduction to Statistical Learning</u> by Witten, James, Hastie, & Tibshirani   
"**MMS**" = <u>Modern Multivariate Statistical Techniques</u> by Izenman  
"**Learning Spark***" = <u>Learning Spark: High Performance Big Data Analysis</u> by Karau, Konwinski, Wendell, and Zaharia  
"**HP Spark***" = <u>High Performance Spark</u> by Karau and Warren  
"**DDS***" = <u>Doing Data Science</u> by O'Neil & Shutt  


*Starred books are ones you will need to purchase or borrow from the UCB library. All other reading materials are open source & linked here for your convenience.

To access the library, go to http://oskicat.berkeley.edu/ and use your Berkeley login.


<table>
<col width="5%">
<col width="30%">
<col width="35%">
<col width="30%">
<tr>
<th></th>
<th>Async</th>
<th>Live Session</th>
<th>Assignment</th>
</tr>
<tr> <!--- WEEK 1 --->
<td width=5%>
<strong>Week 1</strong>
<br>Intro to Machine Learning at Scale
</td>
<td valign="top" width='30%'>   <!-- ASYNCH -->

Read: <a href="http://faculty.marshall.usc.edu/gareth-james/ISL/ISLR%20Seventh%20Printing.pdf">ISL chapter 1 and sections 2.1 &amp; 2.2</a>
<br>
Skim: <a href="https://adamdrake.com/command-line-tools-can-be-235x-faster-than-your-hadoop-cluster.html">Adam Drake Blog Post</a>
<br>
Optional:
<a href="http://scott.fortmann-roe.com/docs/BiasVariance.html" target="_blank">Fortmann-Roe Essay</a>,
<a href="https://theclevermachine.wordpress.com/2013/04/21/model-selection-underfitting-overfitting-and-the-bias-variance-tradeoff/" target="_blank">Clever Machine Blog Post</a>,
<a href="https://insidebigdata.com/2014/10/22/ask-data-scientist-bias-vs-variance-tradeoff/">Inside Big Data Blog Post</a>
</td>
<td valign="top" width='35%'>  <!--- LIVE SESSION --->
<ul>
<li>Course Overview
<li>Bias Variance Tradeoff
<li>Embarrassingly Parallel Problems
<li>Preview of HW 1 &amp; Docker Set-Up
</ul>
<br>
<a href="https://docs.google.com/presentation/d/1bJSkS8KhjarIe9KGq57T7HV1ak-mQhXEuq9riSXeWGA/edit?usp=sharing">[wk 1 Slides]</a>
<br>
<a href="https://github.com/UCB-w261/w261-environment">[Environment Repo]</a>
</td>
<td valign="top" width='30%'> <!-- HOMEWORK --->
<strong><a href="./Assignments/HW1" target="_blank">HW 1</a>:</strong>
<em>This assignment introduces word counting as the "hello-world" of parallel computation. You will implement this task in Python &amp; run it via the command line.</em>
<br>
<br>
<strong>Due 24hrs before your live session 2</strong>
</td>
</tr> <!--- (END) Week 1 --->
<tr> <!--- WEEK 2 --->
<td>
<strong>Week 2</strong>
<br>Parallel Computation Frameworks
</td>
<td valign="top">    <!-- ASYNCH --->

Read: <a href="https://lintool.github.io/MapReduceAlgorithms/MapReduce-book-final.pdf">DITP Chapter 1 &amp; Chapter 2</a>
<br>
Read: <a href="https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf">IIR CH.13</a>
<br>
Optional: <a href="http://www.michael-noll.com/tutorials/writing-an-hadoop-mapreduce-program-in-python/"> Michael Noll Hadoop MR Tutorial</a>
<br>
<br>
<strong>NOTE:</strong>
<em>Please come to class this week with your Docker container running and the data for the demo notebook loaded.</em>
</td>
<td valign="top"> <!--- LIVE SESSION -->
<ul>
<li>Functional Programming
<li>The Map Reduce Paradigm
<li>Intro to Hadoop Streaming Syntax
</ul>
<br>
<a href="https://docs.google.com/presentation/d/1CmVuW89B5Xo_m3QUw9TRVrj8iWsj025qU-MqQMQdHbA/edit#slide=id.g25ebfce285_0_0" >[wk 2 Slides]</a>
<br>
<a href="https://docs.google.com/presentation/d/1_W3hFXoKbCv7UuK7-L4HbOGGFLkWlgXtXOxToR5-jRs/edit?usp=sharing">[Naive Bayes Example Slides]</a>
<br>
<a href="./LiveSessionMaterials/wk02Demo_IntroToHadoop">[wk 2 Demo: Intro to Hadoop]</a>
</td>
<td valign="top"> <!-- HOMEWORK -->
<strong><a href="./Assignments/HW2" target="_blank">HW 2</a>:</strong>
<i> In this assignment you will implement distributed Naive Bayes using Hadoop Streaming.</i>
<br>
<br>
<strong>Due 24hrs before your live session 4</strong>
</td>
</tr> <!--- (END) Week 2 --->
<tr> <!--- WEEK 3 --->
<td>
<strong>Week 3</strong>
<br>MapReduce Algorithm Design
</td>
<td valign="top">    <!-- ASYNCH --->
Read: <a href="https://lintool.github.io/MapReduceAlgorithms/MapReduce-book-final.pdf">DITP sections 2.4 - 2.7 and 3.1-3.4</a>
<br>
Read: <a href="https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf" target="_blank">IIR sections 13.1 and 13.2</a>
<br>
Read: HDG - Part II Chapter 7 - How MapReduce Works   
<br>
<br>
Skim: <a href="./HelpfulResources/TotalSortGuide/_total-sort-guide-spark2.01-JAN27-2017.ipynb">Total Order Sort Guide</a>, <a href="https://patterns.eecs.berkeley.edu/?page_id=37"> EECS Map Reduce Notes</a>
<br>
OPTIONAL: <a href="http://blog.ditullio.fr/category/hadoop-basics/">http://blog.ditullio.fr/category/hadoop-basics/</a>
</td>
<td valign="top"> <!--- LIVE SESSION -->
<ul>
<li> Local Aggregation
<li> Partial vs Total Order Sort
<li> Naive Bayes Parallelization
</ul>
<br>
<a href="https://docs.google.com/presentation/d/1obtsdYckS3yerE_cT9IZfGSOOAxDhGfJsep4rxjzkog/edit#slide=id.g260fdab94d_0_0">[wk 3 Slides]</a>
<br>
<a href="./LiveSessionMaterials/wk03Demo_HadoopShuffle">[wk 3 Demo: Hadoop Shuffle]</a>
<br><br>
</td>
<td valign="top"> <!-- HOMEWORK -->
<strong><a href="./Assignments/HW2" target="_blank">HW 2</a> con't ...</strong> (Due 24hrs before your live session 4)
</td>
</tr> <!--- (END) Week 3 --->
<tr> <!--- WEEK 4 --->
<td>
<strong>Week 4</strong>
<br>Intro to Spark/MapReduce with RDDs
</td>
<td valign="top">   <!-- ASYNCH -->
Read: <a href="https://www.safaribooksonline.com/library/view/high-performance-spark/9781491943199/">HP Spark chapter 2</a>
<br>
Read: <a href="https://spark.apache.org/docs/latest/rdd-programming-guide.html">Spark RDD Programming Guide</a>
<br>
Skim: <a href="https://www.safaribooksonline.com/library/view/learning-spark/9781449359034/">Learning Spark ch 3 &amp; 4</a>
<br>
Skim: <a href="./Readings/wk04-DimensionIndependentSimilarityComputation(Zadeh,Goel).pdf">DISCO Paper</a>
<br>
Skim: <a href="./Readings/wk04-PairwiseDocSim(Elsayed,Lin,Oard).pdf">DocSim Paper</a>  
<br>
  
__Additional Spark resources for weeks 4, and 5__     

* Holden Karau - Spark summit 2017
https://www.youtube.com/watch?v=4xsBQYdHgn8&feature=youtu.be [40 minutes]

* Debugging Spark Holden Karau -Dec 2017
https://www.youtube.com/watch?v=s5p15QT0Zj8&list=WL&index=7&t=0s [45 minutes]


</td>
<td valign="top"> <!--- LIVE SESSION -->
<ul>
	<li>Hadoop Limitations</li>
	<li>Intro to Spark RDDs</li>
	<li>Actions and transformations</li>
	<li>Caching and broadcasting</li>
</ul>
<br>
<a href="https://docs.google.com/presentation/d/1Y8W4WAWcF7ZgXZcRcDtfb-GpIbEWNb7X5lPTm5FpIaA/edit#slide=id.p">[wk 4 Slides]</a>
<br>
<a href="./LiveSessionMaterials/wk04Demo_IntroToSpark">[wk 4 Demo: Intro to Spark]</a>
</td>
<td valign="top"> <!-- HOMEWORK -->
<strong><a href="./Assignments/HW3">HW 3</a></strong>
<em>In this assignment you will perform Synonym Detection in Spark using Google N-gram data.</em>
<br>
<br>
<strong>Due 24hrs before your live session 6</strong>
</td>
</tr> <!--- (END) Week 4 --->
<tr> <!--- WEEK 5 --->
<td>
<strong>Week 5</strong>
<br>Spark/MapReduce with RDDs (con't)
</td>
<td valign="top"> <!-- ASYNCH -->
<strong>Watch:</strong>
<ul>
	<li>
		<a href="https://www.youtube.com/watch?v=s5p15QT0Zj8&list=WL&index=7&t=0s">Debugging Spark - Holden Karau - Dec 2017</a> [45 minutes]
	</li>
	<li>
	Optional: <a href="https://www.youtube.com/watch?v=fp53QhSfQcI">Optimizing Apache Spark SQL Joins: Spark Summit East talk by Vida Ha</a> [30 minutes]
	</li>
</ul>
<br>
<br>
Skim: <a href="./Readings/wk05-IIR(Manning,Raghavan,Schutze)_chapter16.pdf" target="_blank">IIR chapter 16</a>
<br>
Optional: <a href="./Readings/wk05-MMDS-chapter6-Apriori.pdf" target="_blank">Aperiori Algorithm Chapter</a>
</td>
<td valign="top"> <!--- LIVE SESSION -->
<ul> 
	<li> Spark Accumulators</li>
	<li> Custom Partitioning</li>
	<li> Kmeans Algorithm</li>
</ul>
<br>
<a href="https://docs.google.com/presentation/d/1X0azuLULPFjnQhvHqLTiiYuBNrPlDNzd_1TYnZksLU4/edit#slide=id.p">[wk 5 Slides]</a>
<br>
<a href="./LiveSessionMaterials/wk05Demo_Spark/demo5_workbook.ipynb">[wk 5 Demo: K-Means]</a>  
</td>
<td valign="top"> <!-- HOMEWORK -->
<strong><a href="./Assignments/HW3">HW 3</a> con't... </strong>  (Due 24hrs before your live session 6)
</td>
</tr> <!--- (END) Week 5 --->
<tr> <!--- WEEK 6 --->
<td>
<strong>Week 6</strong>
<br>Distributed Supervised ML (part 1)
</td>
<td valign="top"><!-- ASYNCH -->
Read: <a href="https://www-bcf.usc.edu/~gareth/ISL/ISLR%20Seventh%20Printing.pdf">ISL sections 3.1, 3.2</a>
<br>
Skim: <a href="./Readings/wk06-CS273a_Lecture14(DRamanan,UCIrvine).pdf"> UCI cs273a Loss Functions Lecture</a>
</td>
<td valign="top"> <!--- LIVE SESSION -->
<ul>
	<li> Parallelizing Matrix Ops</li>
	<li> Optimization Theory &amp; Gradient Descent</li>
	<li> Linear Regression at Scale</li>
</ul>
<br>
<a href="https://docs.google.com/presentation/d/1wJaU3rCyNa7rhuB5pTL378YTEnaIYFkcQ_B0rU56xa8/edit#slide=id.g27b7ad54fb_0_0">[wk6 Slides]</a>
<br>
<a href="./LiveSessionMaterials/wk06Demo_Optimization">[wk 6 Demo: Gradient Descent]</a>
</td>
<td valign="top"> <!-- HOMEWORK -->
<strong><a href="./Assignments/HW4">HW 4</a></strong>
<em>In this homework you will perform distributed linear regression in Spark.</em>
<br>
<br>
<strong>Due 24hrs before your live session 8</strong>
</td>
</tr> <!--- (END) Week 6 --->
<tr> <!--- WEEK 7 --->
<td>
<strong>Week 7</strong>
<br>Distributed Supervised ML (part 2)
</td>
<td valign="top"><!-- ASYNCH -->
Read:<a href="http://shop.oreilly.com/product/0636920028529.do">DDS chapter 5 (Logistic Regression)</a>
<br>
Read: <a href="https://www-bcf.usc.edu/~gareth/ISL/ISLR%20Seventh%20Printing.pdf">ISL Chapter 4 (Classification), 5.1 (Cross Validation), and 6.1, 6.2 (Shrinkage L1, L2)</a>
<br>
Skim: <a href="./Readings/wk07-MMS(Izenman)_chapter11.pdf" target="_blank"> MMS chapter 11 (SVMs)</a>
<br>
</td>
<td valign="top"> <!--- LIVE SESSION -->
<ul>
	<li> Logistic Regression</li>
	<li> Support Vector Machines</li>
	<li> Regularization</li>
</ul>
<br>
<a href="https://docs.google.com/presentation/d/1Ogz0b6NH0ifARch1mV3aKNt6DiL6chcjgnk4lJUE7Tg/edit#slide=id.g13054ad993ed15f0_0">[wk 7 Slides]</a>
<br>
<a href='./LiveSessionMaterials/wk07Demo_Optimization/demo7_workbook.ipynb'>[Wk7 Demo: Regularization]</a>
</td>
<td valign="top"> <!-- HOMEWORK -->
<strong><a href="./Assignments/HW4">HW 4</a> con't...</strong> (Due 24hrs before your live session 8)
</td>
</tr> <!--- (END) Week 7 --->
<tr> <!--- WEEK 8 --->
<td>
<strong>Week 8</strong>
<br>Big Data Pipelines
</td>
<td valign="top">   <!-- ASYNCH -->
Skim: <a href="https://www.safaribooksonline.com/library/view/high-performance-spark/9781491943199/">HP Spark ch 3-4</a>
<br>
Read: <a href="https://www.safaribooksonline.com/library/view/high-performance-spark/9781491943199/">HP Spark ch 5-6</a>
<br>
Read: <a href="http://www.svds.com/dataformats/">Format Wars Post</a>
<br>
Optional: <a href="https://databricks.com/blog/2016/08/15/how-to-use-sparksession-in-apache-spark-2-0.html">SparkSession article</a>,
<a href="https://sparkour.urizone.net/recipes/understanding-sparksession/">Sparkour recipe</a>
</td>
<td valign="top"> <!--- LIVE SESSION -->
<ul>
	<li> RDDs vs Dataframes</li>
	<li> Compression Methodologies</li>
	<li> Joins and transformations</li>
	<li> Goldilocks Problem</li>
</ul>
<br>
<a href="https://docs.google.com/presentation/d/1cn_2A8UW5b9uuHJ3hicvBLw1BokmvDr7UPEdM3eTlK4/edit?usp=drivesdk">[wk 8 Slides]</a>
<br>
<a href="./LiveSessionMaterials/wk08Demo_DataFrames/demo8_workbook.ipynb">[Week 8 Demo: Advanced Spark - Pipelines and Optimizations with DataFrames]</a>
</td>
<td valign="top"> <!-- HOMEWORK -->
<strong><a href="./Assignments/HW5">HW 5</a></strong>
<em>In this homework you will implement PageRank in Spark.</em>
<br>
<br>
<strong>Due 24hrs before your live session 11</strong>
</td>
</tr> <!--- (END) Week 8 --->
<tr> <!--- WEEK 9 --->
<td>
<strong>Week 9</strong>
<br>Graph Algorithms at Scale (part 1)
</td>
<td valign="top"><!-- ASYNCH -->
Read:<a href="./Readings/wk09-DITP(Lin,Dyer)_chapter5.pdf"> DITP chapter 5</a>
<br>
Skim:<a href="http://www.cs.cornell.edu/courses/cs312/2002sp/lectures/lec20/lec20.htm"> Cornell CS 312 Dijkstra's Lecture</a>
</td>
<td valign="top"> <!--- LIVE SESSION -->
<ul>
<li>Graph Representations for Distributed Processing
<li>BFS, SSSP, and Dijsktra's Algorithm
</ul>
<br>
<a href="https://docs.google.com/presentation/d/1wvZc9Tis8FWmz20Bw2gG7NhnF1XeJQL25hMsPlTEs60/edit#slide=id.g1c0a2650fdc8228_103">[wk 9 Slides]</a>
<br>
<a href="./LiveSessionMaterials/wk09Demo_Graphs">[wk 9 Demo - Distributed Graphs]</a>
</td>
<td valign="top"> <!-- HOMEWORK -->
<strong><a href="./Assignments/HW5">HW 5</a> con't... </strong> (Due 24hrs before your live session 11)
</td>
</tr> <!--- (END) Week 9 --->
<tr> <!--- WEEK 10 --->
<td>
<strong>Week 10</strong>
<br>Graph Algorithms at Scale (part 2)
</td>
<td valign="top"><!-- ASYNCH -->
Read:<a href="./Readings/wk09-DITP(Lin,Dyer)_chapter5.pdf"> DITP chapter 5</a>
<br><br>
<b>OPTIONAL:</b>
<br>	
<a href="https://youtu.be/ZENBQj2qQ2k">Prob & Stats - Markov Chains</a>
<br>
<br>
<b>MIT 6.041 Probabilistic Systems Analysis and Applied Probability</b><br>
<a href="https://www.youtube.com/watch?v=IkbkEtOOC1Y">Lecture 16 - Markov Chains Part 1</a>
<br>
<a href="https://www.youtube.com/watch?v=ZulMqrvP-Pk">Lecture 17 - Markov Chains Part 2</a>   
</td>
<td valign="top"> <!--- LIVE SESSION -->
<ul>
<li> Web Search
<li> Random Walks & Markov Chains
<li> Page Rank
</ul>
<br>
<a href="https://docs.google.com/presentation/d/1MgjY4f8y6K3QFlIFFtsEcEpKMqVhwAKqGGQH8HnLHkM/edit#slide=id.g2aa3b9a546_0_715">[wk 10 Slides]</a>
</td>
<td valign="top"> <!-- HOMEWORK -->
<strong><a href="./Assignments/HW5">HW 5</a> con't... </strong> (Due 24hrs before your live session 11)
</td>
</tr> <!--- (END) Week 10 --->
<tr> <!--- WEEK 11 --->
<td>
<strong>Week 11</strong>
<br>ALS and Spark MLLib
</td>
<td valign="top"><!-- ASYNCH -->
Read: <a href="https://github.com/UCB-w261/main/blob/master/LiveSessionMaterials/wk11Demo_ALS/MatrixFactorization_Koren%2CBell%2CVolinsky.pdf">MATRIX
FACTORIZATION TECHNIQUES FOR RECOMMENDER SYSTEMS</a>	
<br>
Read: <a href="http://shop.oreilly.com/product/0636920028529.do">DDS Chapter 8</a>
<br>
Skim: <a href="https://www.safaribooksonline.com/library/view/learning-spark/9781449359034/">Learning Spark ch 11</a>
</td>
<td valign="top"> <!--- LIVE SESSION -->
<ul>
<li> Collaborative Filtering
<li> Matrix Factorization
<li> Alternating Least Squares
</ul>
<br>
<a href="https://docs.google.com/presentation/d/1OtovseO8WA48btxJpBRAOIc67ctSB0Anu6spZIx30l0">[wk 11 Slides]</a>  
<br>
<a href="./LiveSessionMaterials/wk11Demo_ALS">[wk 11 Demo - ALS in Spark]</a>  
</td>
<td valign="top"> <!-- HOMEWORK -->
Start work on final projects.
</td>
</tr> <!--- (END) Week 11 --->
<tr> <!--- WEEK 12 --->
<td>
<strong>Week 12</strong>
<br>Decision Trees
</td>
<td valign="top"><!-- ASYNCH -->
Read: <a href="https://www-bcf.usc.edu/~gareth/ISL/ISLR%20Seventh%20Printing.pdf">ISL chapter 8</a> (or chapter 9.2 in ESL)
<br>
Read: <a href="https://storage.googleapis.com/pub-tools-public-publication-data/pdf/36296.pdf">PLANET paper</a>	
<br>
<br>
<strong>OPTIONAL</strong>:
<br>
Skim TOC in <a href="https://doc.lagout.org/Others/Data%20Mining/Data%20Mining%20with%20Decision%20Trees_%20Theory%20and%20Applications%20%282nd%20ed.%29%20%5BRokach%20%26%20Maimon%202014-10-23%5D.pdf">DATA MINING WITH DECISION TREES, Theory and Applications</a>	
<br>
A most excellent introduction to Gradient Boosting: <a href="https://explained.ai/gradient-boosting/">How to explain gradient boosting</a>	
</td>
<td valign="top"> <!--- LIVE SESSION -->
<ul>
<li> Bagging and Boosting
<li> Distributed Decision Tree Learning
</ul>
<br>
<a href="https://docs.google.com/presentation/d/1Womuq5YmCNfvRZceguNjettzK0_hh3XojIanrqZ_auQ/edit#slide=id.g289f61603b_0_82">[wk 12 Slides - Decision Trees]</a>
<br>
<a href="./LiveSessionMaterials/wk12Demo_DecisionTrees">[wk 12 Demo - Decision Trees]</a>
</td>
<td valign="top"> <!-- HOMEWORK -->
(n/a)
</td>
</tr> <!--- (END) Week 12 --->
<tr> <!--- WEEK 13 --->
<td>
<strong>Week 13</strong>
<br>Introduction to Streaming Systems
</td>
<td valign="top"><!-- ASYNCH -->
<ul>
	<li>Read: Spark The Definitive Guide Big Data Processing Made Simple - PART V: chapters 20-23</li>
	<li>Read or skim: Chapters 1, 2 - Streaming Systems http://streamingsystems.net/ (these two chapters grew out of these two posts: https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-101, and https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-102) </li>
	<li>Optional: Bennet, Elizabeth. “Putting the Power of Kafka into the Hands of Data Scientists” Stitchfix Company Blog. https://multithreaded.stitchfix.com/blog/2018/09/05/datahighway/</li>
</ul>
</td>
<td valign="top"> <!--- LIVE SESSION -->
<ul>
<li> Data Frames &amp; Spark ML
<li> Spark Streaming Processes
<li> Online CTR prediction
</ul>
<br>
<a href="https://docs.google.com/presentation/d/1PR-qGJVplstpfaGlY70oyetcZZ6NpMPpNlcnwIDPmNM/edit#slide=id.g5dc279309b_0_240">[(New) Streaming Slides]</a>
<br><br>
<a href="https://docs.google.com/presentation/d/1xphJbsUKfaIc_kzdH28BKJVSLiNi77ycbXQM7iC4oZM/edit#slide=id.g39bdd87640_0_154">[Week 13 Slides]</a>
<br>
<a href="./LiveSessionMaterials/wk13Demo_Streaming">[wk 13 Demo - Online Models in Spark]</a>
</td>
<td valign="top"> <!-- HOMEWORK -->
(n/a)
</td>
</tr> <!--- (END) Week 13 --->    
<tr> <!--- WEEK 14 --->
<td>
<strong>Week 14</strong>
<br>Final Project presentations
</td>
<td valign="top"><!-- ASYNCH -->
No new videos this week
<br>
</td>
<td valign="top"> <!--- LIVE SESSION -->
Final Project presentations
</td>
<td valign="top"> <!-- HOMEWORK -->
Final Projects Due 24hrs before your live session.
</td>
</tr> <!--- (END) Week 14 --->   
</table>
