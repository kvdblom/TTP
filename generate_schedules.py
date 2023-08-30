"""Code to generate TTP schedules and check their constraints.

Code adapted from:
https://github.com/kristianverduin/MSc-Thesis-RandomScheduleGeneration-ViolationReduction-TTP/blob/4a00582f02819f7b75a77cd2d638b068026b6608/violations.py
"""

import numpy as np
import random
from datetime import datetime
import sys

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
                             games_played: np.array[int],
                             home_played: np.array[int],
                             away_played: np.array[int]) -> int:
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
    n_rounds = (2*n_teams)-2
    schedule = np.full((n_rounds, n_teams), None)
    choices = list(range(-n_teams, n_teams+1))
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
n_schedules = 1000000

if n_teams % 2 == 0:
    create_schedules(n_teams, n_schedules)
else:
    print(f"Number of teams must be even, but was {n_teams}.")
