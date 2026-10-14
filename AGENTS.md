### Task
Your main tasks when working on is project is:
  - Do code review on the main codes in the project. This includes
     - Planning how to refactor the code in a minimal way that also breaks down the massive SiP class into smaller pieces.

DO NOT modify or add any code yourself unless explicitly prompted. Your default mode of work is coding guidance, review and idea-generation, not writing it yourself. 


Use the .gitignore as a baseline for files that can safely be ignored, unless explicitly asked to work on these. 


### Coding guidelines
- Code related to numerical calculations should be decomposed in functions, whose scope is well defined (has a specific job in some larger script or context) and should preferably not exceed 60 lines including docstring. Furthermore, these should prefer readability over conciseness. Similarly with class methods, these should also have a welldefined purpose, be readable, and preferably be well below 60 lines each. 
- Any time data code is made that writes data to the harddisk, unless there are some good reason not to, the data should be compressed eg. using the numpy.savez_compressed function or the xz tool the situation allows. 
- If at any point you are asked to  modify a file / files, there should be a validation stage where whatever new code is validated against the old. Exceptions to this is if there actually is a mistake which needs correction, in which case the user has to be explicitly notified and a before - after bugfix process. Be sure the user is made aware of any bugs / mistakes.  
- Remember the comments and make sure the code-base is understandable in reasonable time by a human reader. Initial iterations / experimentation of new code can be longer, more ad-hoc, but final versions should have this bullet-point in mind.
