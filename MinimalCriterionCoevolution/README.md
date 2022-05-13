# Minimal Criterion Coevolution

## Maze
### bootstrap
To execute before main mcc process. It prepare initial population.
```
$python bootstrap_maze.py
```
#### options:
- -n --name           : bootstrap name (default: default)
- --agent-num         : number of agent to prepare (default: 20)
- --maze-num          : number of maze to prepare (default: 10)
- --wall-gene-num     : initial number of wall gene (default: 2)
- --path-gene-num     : initial number of path gene (default: 2)
- --num-cores         : number of parallel process (default: 4)

### mcc
```
$python run_maze.py
```
#### options:
- -n --name         : experiment name (default: default)
- -b --bootstrap    : to use bootstrap maze (default: default)
- -g --generation   : number of iteration (default: 1000)
- --agent-criteria  : minimal number to solve maze for each agent to survive (default: 1)
- --maze-criteria   : minimal number of agent to solve for each maze to survive (default: 1)
- --agent-pop       : maximum size of agent population (default: 160)
- --maze-pop        : maximum size of maze population (default: 40)
- --agent-limit     : resource limit of agent (default: 0, it means no limit)
- --maze-limit      : resource limit of maze (default: 4)
- --agent-batch     : batch size of agent (default: 40)
- --maze-batch      : batch size of maze (default: 10)
- --num-cores       : number of parallel process (default: 4)
- --print-maze      : print detail of survived maze genome every generation (default: not)

### make figure
```
$python make_figure.py
```
#### options:
- -n --name         : experiment name
- --num-cores       : number of parallel process (default: 1)
- -cb --colorbar    : plot colorbar of timestep (default: no plot)
- --not-overwrite   : skip process if already gif exists (default: overwrite)
- --no-multi        : do without using multiprocess. if error occur, try this option.