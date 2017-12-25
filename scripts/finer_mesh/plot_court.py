
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle, Rectangle, Arc
from sklearn import mixture
from sklearn import ensemble
from sklearn import cross_validation
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import log_loss
import time
import itertools
import operator


#%% define draw_court() function (from http://savvastjortjoglou.com/nba-shot-sharts.html)

def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    # If an axes object isn't provided to plot onto, just get current one
    if ax is None:
        ax = plt.gca()

    # Create the various parts of an NBA basketball court

    # Create the basketball hoop
    # Diameter of a hoop is 18" so it has a radius of 9", which is a value
    # 7.5 in our coordinate system
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

    # The paint
    # Create the outer box of the paint, width=16ft, height=19ft
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)

    # Three point line
    # Create the side 3pt lines, they are 14ft long before they begin to arc
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    # I just played around with the theta values until they lined up with the
    # threes
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax

def out_teamsDict(team_ID=None, team_abbrev=None):
    out = {}
    teams_json_dat = [
      {
        "teamId": 1610612737,
        "abbreviation": "ATL",
        "teamName": "Atlanta Hawks",
        "simpleName": "Hawks",
        "location": "Atlanta"
      },
      {
        "teamId": 1610612738,
        "abbreviation": "BOS",
        "teamName": "Boston Celtics",
        "simpleName": "Celtics",
        "location": "Boston"
      },
      {
        "teamId": 1610612751,
        "abbreviation": "BKN",
        "teamName": "Brooklyn Nets",
        "simpleName": "Nets",
        "location": "Brooklyn"
      },
      {
        "teamId": 1610612766,
        "abbreviation": "CHA",
        "teamName": "Charlotte Hornets",
        "simpleName": "Hornets",
        "location": "Charlotte"
      },
      {
        "teamId": 1610612741,
        "abbreviation": "CHI",
        "teamName": "Chicago Bulls",
        "simpleName": "Bulls",
        "location": "Chicago"
      },
      {
        "teamId": 1610612739,
        "abbreviation": "CLE",
        "teamName": "Cleveland Cavaliers",
        "simpleName": "Cavaliers",
        "location": "Cleveland"
      },
      {
        "teamId": 1610612742,
        "abbreviation": "DAL",
        "teamName": "Dallas Mavericks",
        "simpleName": "Mavericks",
        "location": "Dallas"
      },
      {
        "teamId": 1610612743,
        "abbreviation": "DEN",
        "teamName": "Denver Nuggets",
        "simpleName": "Nuggets",
        "location": "Denver"
      },
      {
        "teamId": 1610612765,
        "abbreviation": "DET",
        "teamName": "Detroit Pistons",
        "simpleName": "Pistons",
        "location": "Detroit"
      },
      {
        "teamId": 1610612744,
        "abbreviation": "GSW",
        "teamName": "Golden State Warriors",
        "simpleName": "Warriors",
        "location": "Golden State"
      },
      {
        "teamId": 1610612745,
        "abbreviation": "HOU",
        "teamName": "Houston Rockets",
        "simpleName": "Rockets",
        "location": "Houston"
      },
      {
        "teamId": 1610612754,
        "abbreviation": "IND",
        "teamName": "Indiana Pacers",
        "simpleName": "Pacers",
        "location": "Indiana"
      },
      {
        "teamId": 1610612746,
        "abbreviation": "LAC",
        "teamName": "Los Angeles Clippers",
        "simpleName": "Clippers",
        "location": "Los Angeles"
      },
      {
        "teamId": 1610612747,
        "abbreviation": "LAL",
        "teamName": "Los Angeles Lakers",
        "simpleName": "Lakers",
        "location": "Los Angeles"
      },
      {
        "teamId": 1610612763,
        "abbreviation": "MEM",
        "teamName": "Memphis Grizzlies",
        "simpleName": "Grizzlies",
        "location": "Memphis"
      },
      {
        "teamId": 1610612748,
        "abbreviation": "MIA",
        "teamName": "Miami Heat",
        "simpleName": "Heat",
        "location": "Miami"
      },
      {
        "teamId": 1610612749,
        "abbreviation": "MIL",
        "teamName": "Milwaukee Bucks",
        "simpleName": "Bucks",
        "location": "Milwaukee"
      },
      {
        "teamId": 1610612750,
        "abbreviation": "MIN",
        "teamName": "Minnesota Timberwolves",
        "simpleName": "Timberwolves",
        "location": "Minnesota"
      },
      {
        "teamId": 1610612740,
        "abbreviation": "NOP",
        "teamName": "New Orleans Pelicans",
        "simpleName": "Pelicans",
        "location": "New Orleans"
      },
      {
        "teamId": 1610612752,
        "abbreviation": "NYK",
        "teamName": "New York Knicks",
        "simpleName": "Knicks",
        "location": "New York"
      },
      {
        "teamId": 1610612760,
        "abbreviation": "OKC",
        "teamName": "Oklahoma City Thunder",
        "simpleName": "Thunder",
        "location": "Oklahoma City"
      },
      {
        "teamId": 1610612753,
        "abbreviation": "ORL",
        "teamName": "Orlando Magic",
        "simpleName": "Magic",
        "location": "Orlando"
      },
      {
        "teamId": 1610612755,
        "abbreviation": "PHI",
        "teamName": "Philadelphia 76ers",
        "simpleName": "76ers",
        "location": "Philadelphia"
      },
      {
        "teamId": 1610612756,
        "abbreviation": "PHX",
        "teamName": "Phoenix Suns",
        "simpleName": "Suns",
        "location": "Phoenix"
      },
      {
        "teamId": 1610612757,
        "abbreviation": "POR",
        "teamName": "Portland Trail Blazers",
        "simpleName": "Trail Blazers",
        "location": "Portland"
      },
      {
        "teamId": 1610612758,
        "abbreviation": "SAC",
        "teamName": "Sacramento Kings",
        "simpleName": "Kings",
        "location": "Sacramento"
      },
      {
        "teamId": 1610612759,
        "abbreviation": "SAS",
        "teamName": "San Antonio Spurs",
        "simpleName": "Spurs",
        "location": "San Antonio"
      },
      {
        "teamId": 1610612761,
        "abbreviation": "TOR",
        "teamName": "Toronto Raptors",
        "simpleName": "Raptors",
        "location": "Toronto"
      },
      {
        "teamId": 1610612762,
        "abbreviation": "UTA",
        "teamName": "Utah Jazz",
        "simpleName": "Jazz",
        "location": "Utah"
      },
      {
        "teamId": 1610612764,
        "abbreviation": "WAS",
        "teamName": "Washington Wizards",
        "simpleName": "Wizards",
        "location": "Washington"
      }
                     ]
    for team_dict in teams_json_dat:
        out[team_dict["teamId"]] = team_dict
        out[team_dict["abbreviation"]] = team_dict

    if team_ID is not None and team_abbrev is None:
        return out[int(team_ID)]
    elif team_ID is None and team_abbrev is not None:
        return out[team_abbrev]
    else:
        return out
