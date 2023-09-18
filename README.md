# Rules-framework
Framework for selecting segments with reduced target variable content

This framework was created to handle classification tasks when the targeting is 0 or 1. In such tasks, there is sometimes a need to select segments with a reduced content of 0 or 1. It is quite easy to accomplish this task by building 1-2 decision trees, but in such a case, there is a probability that the trees will select too small segments from the initial sample, which is not always acceptable.

Based on the construction of trees, this framework combines branches in various ways (presented in the notebook, and their principles of operation are described) to maximize the size of the final (filtered) sample, with a sufficiently reduced target content.

  Recursive tree traversal algorithms are presented for branch extraction, resulting in lists with usable branches.
  The first algorithm selects each branch separately, while the second algorithm takes all branches that reduce the target content by a sufficient amount from a single tree and produces the best combination of them.
    
  For combining the obtained branches, 2 algorithms are presented:
  
    - random search, for each of the transmitted branches it calculates random combinations, provided that the specified rules on the target content are met
    
    - greedy search, for each of the transmitted branches the greedy algorithm goes through the remaining branches, thus selecting the best combination at the moment.
      (All algorithms are described in more detail in the notebook itself)
