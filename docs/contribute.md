  # Contribution Guidelines
  
  ## Code Guide
  
  ## Code Review
  
  ### How to Merge Pull Request (permissions needed) 
  **Steps**
  
  1. clone, 
  1. inspect, 
  1. compile, 
  1. run tests, 
  1. merge.
  1. and push
    
  **Git Receipe for the steps**
```
   set PullRequest=71; # PR number on github.com
   set branch=operators
   git clone --single-branch -b ${branch} https://github.com/ai-techsystems/dnnCompiler.git
   cd dnnCompiler
   git fetch origin pull/${PullRequest}/head

   git checkout operators
   git merge --no-ff operators
   # resolve conflicts
   # compile and run tests.
   git push origin ${branch}set PullRequest=71; # PR number on github.com
```
  
  ## Document
  
  ## Committer Guide

  #### Forking:  
  * Go to **[dnnCompiler](https://github.com/ai-techsystems/dnnCompiler)**
  * Click **Fork** to your own repository.
    - This will take 10 sec or so.
    - Now you will be redirected to a copy of **dnnCompiler** under your username
    - And it will be written :  
      > your_username/dnnCompiler  
      > forked from ai-techsystems/dnnCompiler
  * Click on the **Clone or Download button** and copy the link.
    - It will look like (https://github.com/your_username/dnnCompiler.git)
  * Go to your terminal and go to any directory under which you want to clone the repo and open terminal.
    - Paste the link you copied after typing `git clone `. It will look like this :  
      ```console
      git clone https://github.com/your_username/dnnCompiler.git
      ```
  #### Changing branch
  - Go inside the repo  
    ```console
    cd dnnCompiler
    ```
  * Now you will be inside the repository.

    - Check how many branches this repository has.
      ```console
      git branch -r
      ``` 
      - You will see something like:
        ```bash
          origin/HEAD -> origin/master
          origin/master
          origin/operators
        ```
    - Check on which branch you are currently on
      ```console
        git branch
      ```
      - You will see something like:
        ```bash
        * master
          operators
        ```
      - The `*` shows your current branch.
    - Change the branch to the operators as all the newer development is done on that branch.
      ```console
        git checkout operators
      ```
      - You will see something like
        ```bash
        Switched to a new branch 'operators'
        Branch 'operators' set up to track remote branch 'operators' from 'origin'.
        ```
    - Now if you do 
      ```console
      git branch
      ```
      - You will see:
        ```bash
          master
        * operators
        ```
      - Now you are on operators branch.
  
  #### Update code  
  * Change the code inside the repo where you want to change.
  #### Backing up uncommitted work:  
  * But first back up your current work:
    ```console
    git stash
    ```
  #### Add synchronization steps to get latest updates from `AITS dnnCompiler`  
  * Now you will have to setup your repo so that it can sync new updates from the original **dnnCompiler** repo under **AITS**. As there will be other developers working on that. To do that you have to set **dnnCompiler** repo of **AITS** as an **upstream**.
  * In the top level under your local **dnnCompiler** repo, open terminal.
    - Add a remote upstream of the original **dnnCompiler** (You only need to do this upstream setup once! But **fetching** and **merging** should be done everytime)
      ```console
      git remote add upstream https://github.com/ai-techsystems/dnnCompiler
      ```
    - This will add original **dnnCompiler** as upstream.
    - To fetch the latest updates from the **dnnCompiler** repo from **AITS**, use
      ```console
      git fetch upstream
      ```
      - You will see something like
        ```bash
        From https://github.com/ai-techsystems/dnnCompiler
        * [new branch]      master     -> upstream/master
        * [new branch]      operators  -> upstream/operators
        ```
  * Now based on which branch you are currently on, you have to merge `origin/branch_name` with `upstream/branch_name`. **Origin** means your forked local repo, and **Upstream** means the original repo from **AITS** here.
  
  #### Merging the update from upstream  
  * If you followed all previous steps, you will be currently on `origin/operators` branch, if you haven't you will be on `origin/master` branch. To check which branch you are on currently, see the above steps. In the next steps, I am assuming you are on `origin/operators` branch.
  * Now we will merge the upstream operators branch.
    ```console
    git merge upstream/operators
    ```
    - This will update your repo with the latest update from upstream repo. If you are already upto date, you will see something like this.
      ```bash
      Already up to date.
      ```
    - Else every update will be merged from operators branch.
  * We will not merge the `upstream/master` as it is not required, but if you want to do that too, follow the steps below.
    - First change to master branch  
      ```console
      git checkout master
      ```  
    - If you did `git fetch` previously, don't bother to do that again, or do a `git fetch upstream`.  
    - Then merge master branch  
      ```console  
      git merge upstream/master
      ```  
    - Now your master branch will also be updated, before you forget, go back to `operators` branch, as we will modify that only.  
      ```console  
      git checkout operators
      ```  
        - Now both of your branches are synchronized with the latest update from **AITS dnnCompiler** repo.  
  * Now your repo is synchronized with the latest update from upstream. Now sync your forked repo with upstream. Till now you synced your local repo with upstream, but not published it in your github forked repo, to do that simply type
    ```console
    git push
    ```
  * Now everything is in sync.
  #### Get uncomitted code back
  * Now get back the local changes you saved earlier with `git stash` command.
    ```console
    git stash pop
    ```
  #### Push your modified code to your forked repo in GitHub
  * Now you will have your uncommitted work over the synced repo, just as you wanted. Do more modifications if required. And then do the usual commands to push your changes in your forked repo.
    ```console
    git add .
    git commit -m "commit message"
    git push
    ```
  * This will update your forked repo with your additions, Now if you want them to be added in the **AITS dnnCompiler** repo, see the Pull request sectionbelow. 
  
  ## Pull Request

  * If you followed previous instructions, you will have a forked repo which has the latest update from **AITS dnnCompiler** with your further modifications.
  * Now go to your forked repo in GitHub in your browser.
  * Change branch from master to operators.
  * You will see something like
    > Your branch is ahead of n commits of ai-techsystems:operators.
  * Click on **pull request**
  * You will be taken to a new page where in the top you can see
    > merge [operator branch] [aits dnnCompiler] <-- [operator branch] [your_username dnnCompiler]
  * You will also be able to see the changes you made in the comparison of files below that.
  * Now click on **create pull request**
  * It's done!  
