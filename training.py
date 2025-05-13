import wandb
import numpy as np
from envs import my_matc_env
from PPO_attention import PPO_Agent
from PPO_attention import Memory
from PPO_attention import compute_gae, discount
from PPO_attention import schedule_beta
import bluesky as bs
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import warnings
from envs.scenario.utils import haversine_distance, normalize
import csv
import os
import time

warnings.filterwarnings("ignore", category=UserWarning, message=".*?.*?")


MAX_NUM_AGENTS = 30
NUM_INTRUDERS = 3

STATE_SIZE = 7
INTRUDER_STATE_SIZE = 6
ACTION_SIZE = 3

NUM_EPISODES = 10000
BATCH_SIZE = 512
EPOCHS = 10
ALPHA = 0.2
BETA = 1e-3
INITIAL_BETA = 0.01
FINAL_BETA = 0.001
LR = 1e-4
POLICY_UPDATE_TIME_STEP = 1024
SHUFFLE = True

TRAINING = True
WANDB = False
RENDER_MODE = 'server'
# Model name
MODEL_NAME = 'best_model_PPO_attention_dummy.pt'


# Example usage
if __name__ == '__main__':

    # Initialize the PPO agent
    state_size = STATE_SIZE 
    action_size = ACTION_SIZE  
    num_intruders = NUM_INTRUDERS  
    batch_size = BATCH_SIZE
    epochs = EPOCHS
    num_agents = MAX_NUM_AGENTS
    numEpisodes = NUM_EPISODES

    # Initialize the memory
    memory = Memory()

    # Initializing the environment
    env = my_matc_env.MATCEnv('routes/matc/routes.json', render_mode=RENDER_MODE)

    # Initialize the PPO agent
    ppo_agent = PPO_Agent(state_size=state_size, action_size=action_size, num_intruders=num_intruders)
    device = ppo_agent.device
    ppo_agent.model.to(device)

    if not TRAINING:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Load the best model
        ppo_agent.load(name=f'{script_dir}/trained_models/best_model_attention_May4.pt')

    # Initialize wandb
    if WANDB:
        wandb.init(project="atc_marl_project3", name="training_run_attention_2_w4_1", config={
            "state_size": state_size,
            "action_size": action_size,
            "num_intruders": num_intruders,
            "num_agents": num_agents,
            "numEpisodes": numEpisodes,
            "batch_size": batch_size,
            "epochs": epochs
        })

    if TRAINING:
        
        best_avg_rewards = float('-inf')  # Initialize the best average rewards
    
    # Training loop
    for episode in range(numEpisodes):

        # Starting time for the episode
        start_time = time.time()

        rewards = {}
        processed_states, infos = env.reset(seed=42)
        episode_actions = {agent: [] for agent in processed_states.keys()}
        # for i in range(episode_length):
        counter = 0
        episodic_losses = []
        entropies = []
        kl_divergences = []
        training_steps = 0
        collided_agents_info = {}

        # update beta 
        beta = schedule_beta(episode=episode, decay_episode=1000, initial_beta=INITIAL_BETA, final_beta=FINAL_BETA)
        # print(f"Beta: {beta}")
        END_EPISODE = False
        while not END_EPISODE: 
            Actions = {}
            LogProbs = {}

            # Prepare batched inputs for all agents
            agent_keys = list(processed_states.keys())

            # Collect all states and contexts as lists (CPU-side)
            all_states = []
            all_contexts = []

            for ownship_agent in agent_keys:
                ownship_state = np.array([value for _, value in processed_states[ownship_agent]['ownship_state'].items()], dtype=np.float32)
                intruder_states = processed_states[ownship_agent]['intruder_states'].astype(np.float32)
                
                all_states.append(ownship_state)
                all_contexts.append(intruder_states)

            # Convert to batched GPU tensors only ONCE
            batched_states = torch.tensor(np.stack(all_states), dtype=torch.float32, device=device)
            batched_contexts = torch.tensor(np.stack(all_contexts), dtype=torch.float32, device=device)

            # Convert batched inputs to PyTorch tensors and transfer to device
            # Ensure all_states and all_contexts have the correct shape
            if len(all_states) == 1:
                batched_states = torch.tensor(all_states[0], dtype=torch.float32, device=device).unsqueeze(0)
            else:
                batched_states = torch.tensor(np.vstack(all_states), dtype=torch.float32, device=device)

            if len(all_contexts) == 1:
                batched_contexts = torch.tensor(all_contexts[0], dtype=torch.float32, device=device).unsqueeze(0)
            else:
                batched_contexts = torch.tensor(np.stack(all_contexts), dtype=torch.float32, device=device)
            
            # Get action probabilities for all agents in a single call
            if TRAINING:
                with torch.no_grad():
                    policies, old_values = ppo_agent.model(batched_states, batched_contexts)
                    dists = torch.distributions.Categorical(policies)
                    actions = dists.sample()
                    logprobs = dists.log_prob(actions)
            else: # Take the action with the highest probability
                with torch.no_grad():
                    policies, old_values = ppo_agent.model(batched_states, batched_contexts)
                    actions = torch.argmax(policies, dim=1)
                    logprobs = torch.zeros_like(actions, dtype=torch.float32).to(device)

            # Assign actions, log probabilities, and old values back to individual agents
            old_values_dict = {}
            for idx, ownship_agent in enumerate(agent_keys):
                Actions[ownship_agent] = actions[idx].item()
                LogProbs[ownship_agent] = logprobs[idx].detach()
                old_values_dict[ownship_agent] = old_values[idx].detach()

            # Step the environment
            next_processed_states, rewards, terminations, truncations, dones = env.step(Actions)
            # print(f"Rewards: {rewards}")

            # Check for collisions
            collided_agents = env.collided_agents
            num_of_collisions = len(env.collided_agents)

            # === [5] Store experience into memory efficiently ===
            for idx, ownship_agent in enumerate(agent_keys):
                reward = rewards[ownship_agent]
                done = dones.get(ownship_agent, 0)
                if reward <= -1:
                    done = 1

                # Use precomputed tensors (slices from the batched input)
                state = batched_states[idx].unsqueeze(0).detach()
                context = batched_contexts[idx].unsqueeze(0).detach()
                action = Actions[ownship_agent]
                logprob = LogProbs[ownship_agent].detach()
                old_value = old_values_dict[ownship_agent].detach()

                # Store into memory
                if reward is not None:
                    memory.add(
                        ownship_agent,
                        state,
                        context,
                        action,
                        reward,
                        done,
                        logprob,
                        old_value
                    )

            for agent in processed_states.keys():
                if agent in episode_actions.keys():
                    episode_actions[agent].append(Actions[agent])
                else:
                    episode_actions[agent] = [Actions[agent]]

            # Update the observations
            processed_states = next_processed_states.copy()

            # Calculate the total length of all states in memory
            total_states_length = sum(len(data['states']) for data in memory.agent_data.values())
            update_time_step = POLICY_UPDATE_TIME_STEP
            # agents_with_large_states = [agent for agent, data in memory.agent_data.items() if len(data['states']) > update_time_step]
            # Increment the counter
            counter += 1

            # collided_agent_list = [item for tup in env.collided_agents for item in tup]

            if TRAINING and len(env.agents) < 1 and len(dones) > (MAX_NUM_AGENTS - 1):

                # print("Training the PPO agent...")
                # Training the PPO agent
                # Check if all agents are done
                all_states, all_contexts, all_actions, all_rewards, all_dones = [], [], [], [], []
                all_advantages, all_returns, all_logprobs = [], [], []

                for agent in memory.agent_data.keys():
                # for agent in truncated_agent_list:
                    # Extract data for the current agent
                    agent_states = torch.stack(memory.agent_data[agent]['states']).to(device)
                    agent_contexts = torch.stack(memory.agent_data[agent]['contexts']).to(device)
                    agent_actions = torch.tensor(memory.agent_data[agent]['actions'], dtype=torch.long).to(device)
                    agent_rewards = torch.tensor(memory.agent_data[agent]['rewards'], dtype=torch.float32).reshape(-1, 1).to(device)  # Added .to(device) here
                    agent_dones = torch.tensor(memory.agent_data[agent]['dones'], dtype=torch.float32).reshape(-1, 1).to(device)  # Added .to(device) here
                    agent_logprobs = torch.tensor(memory.agent_data[agent]['log_probs'], dtype=torch.float32).reshape(-1, 1).to(device)  # Added .to(device) here
                    agent_old_values = torch.stack(memory.agent_data[agent]['old_values']).to(device)  # Use stored old values

                    # Compute GAE and returns for the current agent
                    advantages, returns = compute_gae(agent_rewards, agent_old_values, agent_dones)

                    # Append the current agent's data to the combined lists
                    all_states.append(agent_states)
                    all_contexts.append(agent_contexts)
                    all_actions.append(agent_actions)
                    all_rewards.append(agent_rewards)
                    all_dones.append(agent_dones)
                    all_advantages.append(advantages)
                    all_returns.append(returns)
                    all_logprobs.append(agent_logprobs)                

                states = torch.cat(all_states, dim=0).to(device)
                contexts = torch.cat(all_contexts, dim=0).to(device)
                actions = torch.cat(all_actions, dim=0).to(device)
                rewards = torch.cat(all_rewards, dim=0).to(device)
                dones_ = torch.cat(all_dones, dim=0).to(device)
                advantages = torch.cat(all_advantages, dim=0).to(device)
                returns = torch.cat(all_returns, dim=0).to(device)
                logprobs = torch.cat(all_logprobs, dim=0).to(device)

                # Normalize advantages for the current agent
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Create dataloader for minibatch training
                dataset = TensorDataset(states, contexts, actions, logprobs, returns, advantages)
                loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

                # PPO training loop
                for _ in range(EPOCHS):
                    for s_b, c_b, a_b, old_lp_b, ret_b, adv_b in loader:
                        s_b = s_b.view(-1, STATE_SIZE)
                        c_b = c_b.view(-1, NUM_INTRUDERS, INTRUDER_STATE_SIZE)

                        new_policies, new_values = ppo_agent.model(s_b, c_b)
                        dist = torch.distributions.Categorical(new_policies)

                        old_lp_b = old_lp_b.view(-1)
                        new_logprobs = dist.log_prob(a_b).view(-1)

                        # PPO ratio and losses
                        ratio = (new_logprobs - old_lp_b).exp()
                        surr1 = ratio * adv_b.view(-1)
                        surr2 = torch.clamp(ratio, 1 - ALPHA, 1 + ALPHA) * adv_b.view(-1)
                        policy_loss = -torch.min(surr1, surr2).mean()

                        value_loss = F.mse_loss(new_values, ret_b)
                        entropy = dist.entropy().mean()

                        loss = policy_loss + 0.5 * value_loss - beta * entropy
                        ppo_agent.optimizer.zero_grad()
                        loss.backward() 
                        ppo_agent.optimizer.step()
                        training_steps += 1
                        episodic_losses.append(loss.item())

                        entropies.append(entropy.item())
                        kl_divergences.append((old_lp_b - new_logprobs).mean().item())

                # Reset the memory for the next episode
                memory.reset()

                # Reset the environment
                END_EPISODE = True

            if not TRAINING and len(env.agents) < 1 and len(dones) > (MAX_NUM_AGENTS - 1):
                END_EPISODE = True

            if counter >= MAX_NUM_AGENTS*1000:
                print("Max steps reached. Resetting the environment urgently...")
                END_EPISODE = True
        
        # Update the learning rate
        if TRAINING:
            ppo_agent.update_learning_rate(episode)

        # Calculate the time taken for the episode
        end_time = time.time()
        elapsed_time = end_time - start_time     

        # print out the agent counters from env.agent_counter = {agent: count}
        # print(f"Agent counters: {env.agent_counter}")
        if len(dones) > (MAX_NUM_AGENTS - 1):

            total_steps = sum(env.agent_counter.values())
            if total_steps == 0:
                total_steps = 1
                print("Sth went wrong in the environment. Total steps = 0. Setting it to 1.")

            # collided agents
            collided_agents_info = env.collided_agents

            # finding the sum subrewards of each agent during a given episode
            episode_rewards = env.each_agent_rewards
            sum_episode_rewards = env.sum_episode_rewards

            SumRewards = {}
            for agent, rewards in episode_rewards.items():
                sum_distance_rewards = 0
                sum_velocity_rewards = 0
                sum_action_rewards = 0
                sum_time_rewards = 0
                for reward in rewards:
                    sum_distance_rewards += reward[0]
                    sum_velocity_rewards += reward[1]
                    sum_action_rewards += reward[2]
                    sum_time_rewards += reward[3]
                SumRewards[agent] = [sum_distance_rewards, sum_velocity_rewards, sum_action_rewards, sum_time_rewards]

            sum_rewards_all_agents = 0
            sum_distance_rewards_all_agents = 0
            sum_velocity_rewards_all_agents = 0
            sum_action_rewards_all_agents = 0
            sum_time_rewards_all_agents = 0
            for agent in SumRewards.keys():
                sum_distance_rewards_all_agents += SumRewards[agent][0]
                sum_velocity_rewards_all_agents += SumRewards[agent][1]
                sum_action_rewards_all_agents += SumRewards[agent][2]
                sum_time_rewards_all_agents += SumRewards[agent][3]

            avg_distance_rewards_all_agents = sum_distance_rewards_all_agents / total_steps * counter
            avg_velocity_rewards_all_agents = sum_velocity_rewards_all_agents / total_steps * counter
            avg_action_rewards_all_agents = sum_action_rewards_all_agents / total_steps * counter
            avg_time_rewards_all_agents = sum_time_rewards_all_agents / total_steps * counter
            sum_rewards_all_agents = sum_episode_rewards #sum_distance_rewards_all_agents + sum_velocity_rewards_all_agents #+ sum_action_rewards_all_agents + sum_time_rewards_all_agents

            avg_rewards_all_agents = sum_episode_rewards/total_steps * counter
            # avg_rewards_all_agents = sum_episode_rewards/counter * 100.0

            # Save the best model after 100 episodes
            script_dir = os.path.dirname(os.path.abspath(__file__))
            if TRAINING and episode % 100 == 0:
                if avg_rewards_all_agents > best_avg_rewards:
                    best_avg_rewards = avg_rewards_all_agents
                    print(f"New best average rewards: {best_avg_rewards}. Saving the model...")
                    
                    ppo_agent.save(save_path=f'{script_dir}/trained_models/{MODEL_NAME}')  # Save the best model

            avg_episode_actions = {agent: float(np.mean(actions)) for agent, actions in episode_actions.items()}

            # Average of avg_episode_actions
            avg_action = np.mean(list(avg_episode_actions.values()))

            # Average loss per episode
            avg_loss = np.mean(episodic_losses)

            # Average entropy per episode
            avg_entropy = np.mean(entropies)
            avg_kl_divergence = np.mean(kl_divergences)

            # Format `sum_rewards` to 4 decimal places
            avg_rewards_formatted = f"{avg_rewards_all_agents:.4f}"

            # Format `SumRewards` dictionary values to 4 decimal places
            # SumRewards_formatted = {agent: [f"{reward:.4f}" for reward in rewards] for agent, rewards in SumRewards.items()}
            sum_subrewards_formatted = [f"{sum_distance_rewards_all_agents:.4f}", f"{sum_action_rewards_all_agents:.4f}", f"{sum_time_rewards_all_agents:.4f}"]
            avg_subrewards_formatted = [f"{avg_distance_rewards_all_agents:.4f}", f"{avg_action_rewards_all_agents:.4f}", f"{avg_time_rewards_all_agents:.4f}"]
            # Format `avg_episode_actions` dictionary values to 4 decimal places
            avg_episode_actions_formatted = {agent: f"{avg_action:.4f}" for agent, avg_action in avg_episode_actions.items()}

            # Foramt avg_loss to 4 decimal places
            avg_loss = f"{avg_loss:.4f}"

            # Print the formatted results
            print(f"\nEpisode {episode}>> Collisions: {num_of_collisions}, Avg Rewards: {avg_rewards_formatted}, Avg Subrewards: {avg_subrewards_formatted}, Sum Subrewards: {sum_subrewards_formatted}, Avg Loss: {avg_loss}, Elapsed Time: {elapsed_time:.2f}, LR: {ppo_agent.new_lr:.5f}, Beta: {beta:.4f}, Training Steps: {training_steps}.")

            # print(f"Average Actions: {avg_episode_actions_formatted}.")

            # print(f"Episode Counter:{counter}, Collided Agents: {collided_agents_info}, Average Actions: {avg_episode_actions_formatted}.")

            if WANDB:
                wandb.log({
                    "Average Rewards": avg_rewards_all_agents,
                    "Sum Distance Rewards": sum_distance_rewards_all_agents,
                    "Sum Action Rewards": sum_action_rewards_all_agents,
                    "Sum Time Rewards": sum_time_rewards_all_agents,
                    "Collisions": num_of_collisions,
                    "Average Actions": avg_action,
                    "entropy": avg_entropy,
                    "KL Divergence": avg_kl_divergence,
                    "Average Loss": float(avg_loss),
                    "Learning Rate": ppo_agent.new_lr,
                    "Entropy Rate": beta
                })

        else:
            print(f"Episode {episode}: No agents left in the environment. Ending episode.")
            break


    # Finish wandb run
    if WANDB:
        wandb.finish()



