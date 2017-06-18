# TicTacToe-RL

Tic-tac-toe solver to illustrate Q Learning

Coded by Lyote, dusted down by p-i-, June 2017

Q (and Deep-Q) Learning primer at https://www.nervanasys.com/demystifying-deep-reinforcement-learning/

![](TicTac.png?raw=true)

At the heart of this algorithm is a Q-table of State+Action -> Reward. Each (state, action) pair has an associated reward. e.g. The action of placing an 'X' at position 8 (bottom right corner) on the board-state '--X OOX ---' might have an associated reward of +1, as it is an immediate win.

Initially the table is empty; it is written to at the end of each training (computer-computer) episode/round.

For each move in the episode, we examine the associated reward: `Q_table[state, action]` and nudge it in the direction of an improved calculation.

The improvement is: consider the new state `s'` we arrive at by performing the action. Figure out (using the Q-table) the best (highest reward) action one can take from `s'`.  So the improved calculation for the gross-reward at `s` is the immediate reward of `s -> s'` + `lambda` * the gross-reward at `s'`. Let's call that `R_new`.

So we nudge `R` in the direction of `R_new`: `R  =  0.9 * R  +  0.1 * R_new`

There is one problem with this: what if we try to access/modify a `Q_table` entry that does not exist yet? 

    # Get Q-value for a particular action on a given board(-state)
    def Q_read(self, nAction, state):
        # type: (int, BoardState) -> float
        return self.Q_tables[nAction].get(state, self.default_Q)

So it produces the value `default_Q` if the entry does not exist.

The only values we attempt to modify are the `(state, action)` pairs that get produced by playing through an episode:

    while True:
        action = agent.choose_action(state)
        state, reward, done = state.step(action)
        history.append((action, reward, state))
        if done:
            break

Observe `BoardState.step()` -- which applies an action to a board, returning a reward of `+1` if it results in a win, `-1` if it is an illegal move (i.e. trying to mark a nonempty square), otherwise `0`.

So when we train our Q-table with this history, we have (state, action) -> immediate_reward (got from `step()`) -- so we will nudge whatever value our Q-table currently has for this (state, action) pair towards immediate_reward + lambda * maximum-reward (as calculated Q-table) at the new state we arrive at.
