python -c "import torch; print(torch.cuda.is_available())"
python -c "import jax; print(jax.devices())"

KeyArray -> PRNGKey
jnp.ndarray = jnp.array([]) -> jnp.ndarray = flax.struct.field(default_factory=lambda: jnp.array([...]))

For OGbench cube-quadruple
--ogbench_dataset_dir=/data/cyh/ogbench/cube-quadruple-play-100m-v0

Format On Save

# QC
## D4RL
### antmaze
#### antmaze-medium-diverse-v2
python main.py --agent.actor_type=best-of-n --agent.actor_num_samples=32 --env_name=antmaze-medium-diverse-v2 --sparse=True --horizon_length=5 --exp_name=antmaze-medium-diverse-v2_QC
#### antmaze-large-diverse-v2
python main.py --agent.actor_type=best-of-n --agent.actor_num_samples=32 --env_name=antmaze-large-diverse-v2 --sparse=True --horizon_length=5 --exp_name=antmaze-large-diverse-v2_QC
### Adroit
#### pen-human-v1
python main.py --agent.actor_type=best-of-n --agent.actor_num_samples=32 --env_name=pen-human-v1 --sparse=True --horizon_length=5 --exp_name=pen-human-v1_QC
## OGbench 
### puzzle-3x3-play
#### task4-v0
python main.py  --agent.actor_type=best-of-n --agent.actor_num_samples=32 --env_name=puzzle-3x3-play-singletask-task4-v0 --sparse=True --horizon_length=5 --exp_name=puzzle-3x3-play-singletask-task4-v0_QC
### scene
#### task2-v0
python main.py  --agent.actor_type=best-of-n --agent.actor_num_samples=32 --env_name=scene-play-singletask-task2-v0 --sparse=False --horizon_length=5 --exp_name=scene-play-singletask-task2-v0_QC
### cube-double
#### task2-v0
python main.py  --agent.actor_type=best-of-n --agent.actor_num_samples=32 --env_name=cube-double-play-singletask-task2-v0 --sparse=False --horizon_length=5 --exp_name=cube-double-play-singletask-task2-v0_QC
### cube-triple
#### task2-v0
python main.py  --agent.actor_type=best-of-n --agent.actor_num_samples=32 --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5
## Robomimic 
### lift
python main.py  --agent.actor_type=best-of-n --agent.actor_num_samples=32 --env_name=lift-mh-low --sparse=False --horizon_length=5 --exp_name=lift-mh-low-task2-v0_QC
### can
python main.py --agent.actor_type=best-of-n --agent.actor_num_samples=32 --env_name=can-mh-low --sparse=False --horizon_length=5
### square
python main.py --agent.actor_type=best-of-n --agent.actor_num_samples=32 --env_name=square-mh-low --sparse=False --horizon_length=5
### transport
python main.py --agent.actor_type=best-of-n --agent.actor_num_samples=32 --env_name=transport-mh-low --sparse=False --horizon_length=5 --exp_name=transport-mh-low-task2-v0_QC

# QC-FQL
## D4RL
### antmaze
#### antmaze-medium-diverse-v2
python main.py --agent.alpha=100 --env_name=antmaze-medium-diverse-v2 --sparse=True --horizon_length=5 --exp_name=antmaze-medium-diverse-v2_QC-FQL
#### antmaze-large-diverse-v2
python main.py --agent.alpha=100 --env_name=antmaze-large-diverse-v2 --sparse=True --horizon_length=5 --exp_name=antmaze-large-diverse-v2_QC-FQL
### Adroit
#### pen-human-v1
python main.py --agent.alpha=100 --env_name=pen-human-v1 --sparse=True --horizon_length=5 --exp_name=pen-human-v1_QC-FQL
## OGbench 
### puzzle-3x3-play
#### task4-v0
python main.py --agent.alpha=100 --env_name=puzzle-3x3-play-singletask-task4-v0 --sparse=True --horizon_length=5 --exp_name=puzzle-3x3-play-singletask-task4-v0_QC-FQL
### scene
#### task2-v0
python main.py --agent.alpha=100 --env_name=scene-play-singletask-task2-v0 --sparse=False --horizon_length=5 --exp_name=scene-play-singletask-task2-v0_QC-FQL
### cube-double
#### task2-v0
python main.py --agent.alpha=100 --env_name=cube-double-play-singletask-task2-v0 --sparse=False --horizon_length=5 --exp_name=cube-double-play-singletask-task2-v0_QC-FQL
### cube-triple
#### task1-v0
python main.py --agent.alpha=100 --env_name=cube-triple-play-singletask-task1-v0 --sparse=False --horizon_length=5 --exp_name=cube-triple-play-singletask-task1-v0_QC-FQL
#### task2-v0
python main.py --agent.alpha=100 --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5 --exp_name=cube-triple-play-singletask-task2-v0_QC-FQL
## Robomimic 
### lift
python main.py --agent.alpha=100 --env_name=lift-mh-low --sparse=False --horizon_length=5 --exp_name=lift-mh-low-task2-v0_QC-FQL
### can
python main.py --agent.alpha=100 --env_name=can-mh-low --sparse=False --horizon_length=5
### square
python main.py --agent.alpha=100 --env_name=square-mh-low --sparse=False --horizon_length=5
### transport
python main.py --agent.alpha=100 --env_name=transport-mh-low --sparse=False --horizon_length=5 --exp_name=transport-mh-low-task2-v0_QC-FQL

# QC-FQL-MPO
## D4RL
### antmaze
#### antmaze-medium-diverse-v2
python main.py --agent=agents/acmpo.py --agent.alpha=100 --env_name=antmaze-medium-diverse-v2 --sparse=True --horizon_length=5 --exp_name=antmaze-medium-diverse-v2_QC-FQL-MPO_normweight-0.1-meansum 
#### antmaze-large-diverse-v2
python main.py --agent=agents/acmpo.py --agent.alpha=100 --env_name=antmaze-large-diverse-v2 --sparse=True --horizon_length=5 --exp_name=antmaze-large-diverse-v2_QC-FQL-MPO_normweight-0.1-meansum 
### Adroit
#### pen-human-v1
python main.py --agent=agents/acmpo.py --agent.alpha=100 --env_name=pen-human-v1 --sparse=True --horizon_length=5 --exp_name=pen-human-v1_QC-FQL-MPO_normweight-0.1-meansum 
## OGbench 
### puzzle-3x3-play
#### task4-v0
python main.py --agent=agents/acmpo.py --agent.alpha=100 --env_name=puzzle-3x3-play-singletask-task4-v0 --sparse=True --horizon_length=5 --exp_name=puzzle-3x3-play-singletask-task4-v0_QC-FQL-MPO_normweight-0.1-meansum 
### scene
#### task2-v0
python main.py --agent=agents/acmpo.py --agent.alpha=100 --env_name=scene-play-singletask-task2-v0 --sparse=False --horizon_length=5 --exp_name=scene-play-singletask-task2-v0_QC-FQL-MPO_normweight-1-meansum 
### cube-double
#### task2-v0
python main.py --agent=agents/acmpo.py --agent.alpha=100 --env_name=cube-double-play-singletask-task2-v0 --sparse=False --horizon_length=5 --exp_name=cube-double-play-singletask-task2-v0_QC-FQL-MPO_normweight-1-meansum
### cube-triple
#### task1-v0
python main.py --agent=agents/acmpo.py --agent.alpha=100 --env_name=cube-triple-play-singletask-task1-v0 --sparse=False --horizon_length=5 --exp_name=cube-triple-play-singletask-task1-v0_QC-FQL-MPO_normweight-1-meansum
#### task2-v0
python main.py --agent=agents/acmpo.py --agent.alpha=100 --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5 --exp_name=cube-triple-play-singletask-task2-v0_QC-FQL-MPO_normweight-1-meansum
## Robomimic
### lift
python main.py --agent=agents/acmpo.py --agent.alpha=100 --env_name=lift-mh-low --sparse=False --horizon_length=5 --exp_name=lift_QC-FQL-MPO_normweight-0.1-meansum
### can
python main.py  --agent=agents/acmpo.py --agent.alpha=100 --env_name=can-mh-low --sparse=False --horizon_length=5 --exp_name=can_QC-FQL-MPO_normweight-0.1-meansum
### square
python main.py  --agent=agents/acmpo.py --agent.alpha=100 --env_name=square-mh-low --sparse=False --horizon_length=5 --exp_name=square_QC-FQL-MPO_normweight-0.1
### transport
python main.py  --agent=agents/acmpo.py --agent.alpha=100 --env_name=transport-mh-low --sparse=False --horizon_length=5 --exp_name=transport_QC-FQL-MPO_normweight-0.1-meansum