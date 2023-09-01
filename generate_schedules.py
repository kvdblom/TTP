"""Code to generate TTP schedules and check their constraints.

Code adapted from:
https://github.com/kristianverduin/MSc-Thesis-RandomScheduleGeneration-ViolationReduction-TTP/blob/4a00582f02819f7b75a77cd2d638b068026b6608/violations.py
"""

import random
from datetime import datetime
import sys

import numpy as np
import pandas as pd

max_streak = 3


def check_schedule_constraints(schedule: np.array, n_teams: int) -> list[int]:
    """Calculate the number of violations present in the schedule.

    Arguments:
        schedule ([int, int]) : Schedule
        n_teams (int) : The number of teams present in the schedule

    Returns:
        violations ([int]) : The number of violations present in the schedule,
        in the format [Home/Away streak, No-repeat, Double round-robin,
        mismatched games, games against itself]
    """
    n_rounds = (2 * n_teams) - 2
    violations = [0, 0, 0, 0, 0]

    for team in range(n_teams):
        home_streak = 0
        away_streak = 0
        games_played = np.zeros(n_teams)
        home_played = np.zeros(n_teams)
        away_played = np.zeros(n_teams)

        for round_n in range(n_rounds):
            # Check maxStreak
            if schedule[round_n, team] > 0:
                away_streak = 0
                home_streak += 1
                home_played[abs(schedule[round_n, team])-1] += 1
            else:
                away_streak += 1
                home_streak = 0
                away_played[abs(schedule[round_n, team])-1] += 1
            if home_streak > max_streak or away_streak > max_streak:
                violations[0] += 1

            games_played[abs(schedule[round_n, team])-1] += 1

            # Check noRepeat
            if round_n > 0:
                if (abs(schedule[round_n, team])
                        == abs(schedule[round_n-1, team])):
                    violations[1] += 1

            # Check if the opponent also has the current team as opponent
            # (matches are paired)
            violations[3] += check_mismatched_games(team, round_n, schedule)

            # Check if the current team is playing against itself
            if abs(schedule[round_n, team]) - 1 == team:
                violations[4] += 1

        # Check for double round-robin violations
        violations[2] += check_double_round_robin(
            team, games_played, home_played, away_played)

    return violations


def check_double_round_robin(team: int,
                             games_played: np.array,
                             home_played: np.array,
                             away_played: np.array) -> int:
    """Check for double round-robin violations.

    Arguments:
        team: int indicating which team to check
        games_played: int array with total games played per opponent
        home_played: int array with games played at home per opponent
        away_played: int array with games played away per opponent

    Returns:
        0 if no double round-robin violations, positive int otherwise.
    """
    violations = 0

    for i in [teams for teams in range(len(games_played)) if teams != team]:
        if games_played[i] == 0:
            violations += 2  # Two violations, because two games too few
        elif games_played[i] == 1:
            violations += 1
        elif games_played[i] == 2:
            if home_played[i] != 1 and away_played[i] != 1:
                violations += 1
        else:
            violations += games_played[i] - 2

            if home_played[i] == 0 or away_played[i] == 0:
                violations += 1

    return violations


def check_mismatched_games(team: int, round_n: int, schedule: np.array) -> int:
    """Check if the opponent also has the current team as opponent.

    (matches are paired)

    Arguments:
        team: int indicating which team to check
        round_n: int indicating which round to check
        schedule ([int, int]) : Schedule

    Returns:
        0 if the game is not mismatched, 1 if it is mismatched.
    """
    violations = 0

    if team != abs(schedule[round_n, abs(schedule[round_n, team])-1])-1:
        violations = 1

    return violations


def create_random_schedule_pairs(n_teams: int) -> np.array:
    """Generate a randomly paired schedule.

    Arguments:
        n_teams (int) : The number of teams present in the schedule

    Returns:
        Schedule ([int, int]) : The randomly generated schedule
    """
    n_rounds = (2 * n_teams) - 2
    schedule = np.full((n_rounds, n_teams), None)
    choices = list(range(-n_teams, n_teams + 1))
    choices.remove(0)

    for round_n in range(n_rounds):
        teams_to_pick = choices.copy()

        for team in range(n_teams):
            if schedule[round_n, team] is None:
                team += 1
                teams_to_pick.remove(team)
                teams_to_pick.remove(-team)
                choice = random.choice(teams_to_pick)
                teams_to_pick.remove(choice)
                teams_to_pick.remove(-choice)

                if choice > 0:
                    schedule[round_n, team-1] = choice
                    schedule[round_n, choice-1] = -team
                else:
                    schedule[round_n, team-1] = choice
                    schedule[round_n, abs(choice)-1] = team

    return schedule


def create_random_schedule_games(n_teams: int) -> (np.array, bool):
    """Generate a schedule by randomly assigning feasible games.

    Arguments:
        n_teams (int) : The number of teams present in the schedule

    Returns:
        Tuple with:
            - Schedule ([int, int]) : The randomly generated schedule
            - True if the schedule is feasible, False if the schedule is
            infeasible (and therefore incomplete)
    """
    n_rounds = (2 * n_teams) - 2
    schedule = np.full((n_rounds, n_teams), None)
    games = generate_games_list(n_teams)

    for round_n in range(n_rounds):
        # Take all games still awaiting assignment
        games_to_pick = games.copy()

        for team in range(n_teams):
            if schedule[round_n, team] is None:
                team += 1
                # 1. Take all games relevant for this team and round, that
                # don't violate the constraints.
                games_relevant = games_to_pick.copy()
                # a. Remove games that don't involve this team
                games_relevant.drop(
                    games_relevant[
                        (games_relevant.home_teams != team)
                        & (games_relevant.away_teams != team)].index,
                    axis=0, inplace=True)

                # b. Remove games that repeat the opponent from last round
                if round_n > 0:
                    games_relevant.drop(
                        games_relevant[
                            (games_relevant.home_teams
                             == abs(schedule[round_n-1, team-1]))
                            | (games_relevant.away_teams
                               == abs(schedule[round_n-1, team-1]))].index,
                        inplace=True)

                # c. If max_streak would be exceeded, remove home/away games
                # involving that team, depending on which streak it is.
                for player in range(n_teams):
                    streak_home, streak_away = count_streak(
                        schedule, player, round_n)

                    if streak_home >= max_streak:
                        games_relevant.drop(
                            games_relevant[
                                games_relevant.home_teams == player].index,
                            inplace=True)
                    elif streak_away >= max_streak:
                        games_relevant.drop(
                            games_relevant[
                                games_relevant.away_teams == player].index,
                            inplace=True)

                # 2. Try to assign a game
                # a. Check if any options are available, otherwise return early
                if len(games_relevant.index) == 0:
                    return (schedule, False)

                # b. Select the game
                choice = games_relevant.sample().reset_index(drop=True)
                choice_home = choice.at[0, "home_teams"]
                choice_away = choice.at[0, "away_teams"]

                # c. Put game in schedule, indicating away with negative
                schedule[round_n, choice.home_teams-1] = -choice.away_teams
                schedule[round_n, choice.away_teams-1] = choice.home_teams

                # 3. Remove games from relevant DataFrames to prepare for the
                # next assignment and/or round.
                # a. Remove the choice from the games DataFrame
                games.drop(
                    games[(games.home_teams == choice_home)
                          & (games.away_teams == choice_away)].index,
                    inplace=True)

                # b. Remove all games involving the teams of the choice from
                # the games_to_pick DataFrame (they cannot be placed again in
                # this round)
                games_to_pick.drop(
                    games_to_pick[
                        (games_to_pick.home_teams == choice_home)
                        | (games_to_pick.away_teams == choice_home)
                        | (games_to_pick.home_teams == choice_away)
                        | (games_to_pick.away_teams == choice_away)].index,
                    inplace=True)

    return (schedule, True)


def count_streak(schedule: np.array,
                 team: int,
                 current_round: int) -> (int, int):
    """Count the home and away streaks of a team for an incomplete schedule.

    Arguments:
        schedule: A partial schedule.
        team: The team to count the streaks for.
        current_round: The round for which to count the preceding streaks.

    Returns:
        streak_home: int of the home streak
        streak_away: int of the away streak
    """
    streak_home = 0
    streak_away = 0

    for round_n in range(current_round - 1, -1, -1):
        if schedule[round_n, team-1] is None:
            # If None is found the streak ends here and we stop
            # NOTE: Could lead to infeasible assignment if relied on when
            # assigning to random slots in the schedule, rather than from the
            # top
            break
        elif schedule[round_n, team-1] > 0 and streak_away == 0:
            streak_home += 1
        elif schedule[round_n, team-1] < 0 and streak_home == 0:
            streak_away += 1
        else:
            # Either a home or away game was in the schedule, while the other
            # streak was non-zero, so we reached a switching point and stop.
            break

    return (streak_home, streak_away)


def generate_games_list(n_teams: int) -> pd.DataFrame:
    """Generate a list of games for the given number of teams.

    Arguments:
        n_teams: The number of teams in the tournament.

    Returns:
        pd.DataFrame with two columns:
            home_teams: int of the team number
            away_teams: int of the team number
    """
    home_teams = list()
    away_teams = list()

    for home_team in range(n_teams):
        for away_team in [teams for teams in range(n_teams)
                          if teams != home_team]:
            home_teams.append(home_team + 1)
            away_teams.append(away_team + 1)

    games = pd.DataFrame({"home_teams": home_teams,
                          "away_teams": away_teams})

    return games


def create_schedules(n_teams: int, n_schedules: int) -> None:
    """Generate n schedules and saves the violations of the schedules.

    Arguments:
        n_schedules (int) : The number of schedules to generate
        n_teams (int) : The number of teams present in the schedule

    Returns:
        TotalViolations ([int, int]): The violations present in all n schedules
    """
    start = datetime.now()

    home_away_min = []
    repeat_min = []
    robin_min = []
    total = []

    for i in range(n_schedules):
        schedule = create_random_schedule_pairs(n_teams)
        violations = check_schedule_constraints(schedule, n_teams)

        home_away = violations[0]
        repeat = violations[1]
        robin = violations[2]

        value = np.sum(violations)

        if i == 0:
            home_away_min.append(home_away)
            repeat_min.append(repeat)
            robin_min.append(robin)
            total.append(value)
        else:
            last_value = (
                home_away_min[len(home_away_min)-1]
                + repeat_min[len(repeat_min)-1]
                + robin_min[len(robin_min)-1])
            if value <= last_value:
                home_away_min.append(home_away)
                repeat_min.append(repeat)
                robin_min.append(robin)
                total.append(value)
            else:
                home_away_min.append(home_away_min[len(home_away_min)-1])
                repeat_min.append(repeat_min[len(repeat_min)-1])
                robin_min.append(robin_min[len(robin_min)-1])
                total.append(last_value)

    time = datetime.now() - start
    time = time.total_seconds()


n_teams = int(sys.argv[1])
n_schedules = 100000

if n_teams % 2 == 0:
    perfect = 0
    for i in range(0, n_schedules):
        schedule, good = create_random_schedule_games(n_teams)
        if good:
            # print(schedule)
            constraints = check_schedule_constraints(schedule, n_teams)
            print(f"{i}: {constraints}")
            if sum(constraints) == 0:
                perfect += 1
    print(
        f"n_teams: {n_teams}, n_schedules: {n_schedules}, perfect: {perfect}")
    # create_schedules(n_teams, n_schedules)
else:
    print(f"Number of teams must be even, but was {n_teams}.")
