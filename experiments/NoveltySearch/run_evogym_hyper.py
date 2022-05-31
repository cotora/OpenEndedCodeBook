import sys
import os
import numpy as np

import evogym.envs


CURR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURR_DIR))

LIB_DIR = os.path.join(ROOT_DIR, 'libs')
sys.path.append(LIB_DIR)
import ns_neat
from parallel import ParallelEvaluator
from experiment_utils import initialize_experiment

ENV_DIR = os.path.join(ROOT_DIR, 'envs', 'evogym')
sys.path.append(ENV_DIR)
from evaluator import EvogymControllerEvaluatorNS
from simulator import EvogymControllerSimulator, SimulateProcess
from cppn_decoder import EvogymHyperDecoder
from substrate import Substrate
from gym_utils import load_robot


from arguments.evogym_ns_hyper import get_args


def main():
    args = get_args()

    save_path = os.path.join(CURR_DIR, 'out', 'evogym_ns_hyper', args.name)

    initialize_experiment(args.name, save_path, args)


    structure = load_robot(ROOT_DIR, args.robot, task=args.task)


    substrate = Substrate(args.task, structure[0])
    decoder = EvogymHyperDecoder(substrate, use_hidden=args.use_hidden)
    decode_function = decoder.decode

    evaluator = EvogymControllerEvaluatorNS(args.task, structure, args.eval_num)
    evaluate_function = evaluator.evaluate_controller

    parallel = ParallelEvaluator(
        num_workers=args.num_cores,
        evaluate_function=evaluate_function,
        decode_function=decode_function
    )


    config_file = os.path.join(CURR_DIR, 'config', 'evogym_ns_hyper.cfg')
    custom_config = [
        ('NS-NEAT', 'pop_size', args.pop_size),
        ('NS-NEAT', 'metric', 'manhattan'),
        ('NS-NEAT', 'threshold_init', args.ns_threshold),
        ('NS-NEAT', 'threshold_floor', 0.001),
        ('NS-NEAT', 'neighbors', args.num_knn),
        ('NS-NEAT', 'mcns', args.mcns),
        ('DefaultGenome', 'num_inputs', decoder.input_dims),
        ('DefaultGenome', 'num_outputs', decoder.output_dims)
    ]
    config = ns_neat.make_config(config_file, custom_config=custom_config)
    config_out_file = os.path.join(save_path, 'evogym_ns_hyper.cfg')
    config.save(config_out_file)


    pop = ns_neat.Population(config)

    reporters = [
        ns_neat.SaveResultReporter(save_path),
        ns_neat.NoveltySearchReporter(True),
    ]
    for reporter in reporters:
        pop.add_reporter(reporter)


    if not args.no_view:
        simulator = EvogymControllerSimulator(
            env_id=args.task,
            structure=structure,
            decode_function=decode_function,
            load_path=save_path,
            history_file='history_reward.csv',
            genome_config=config.genome_config)

        simulate_process = SimulateProcess(
            simulator=simulator,
            generations=args.generation)

        simulate_process.init_process()
        simulate_process.start()


    pop.run(evaluate_function=parallel.evaluate, n=args.generation)

if __name__=='__main__':
    main()
