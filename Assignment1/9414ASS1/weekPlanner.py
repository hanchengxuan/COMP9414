# It's like painting a runway with different paints

from cspProblem import CSP, Constraint
from display import Displayable
import sys
from collections import defaultdict


def time_transfer(time):  # time format change
    if time % 24 != 0:
        if int(time % 24) > 12:
            ans = str(day_num_reverse[str(time // 24)]) + " " + str((time % 24) - 12) + 'pm'
        elif int(time % 24) < 12:
            ans = str(day_num_reverse[str(time // 24)]) + " " + str(time % 24) + 'am'
        else:
            ans = str(day_num_reverse[str(time // 24)]) + " " + str(time % 24) + 'pm'
    return ans


def compute_actual_time(time):
    while time > 24:
        time -= 24
    return time

# implementation of weekplanner CSP object


class CSP_1(CSP):
    def __init__(self, domains, constraints, activity):
        """pass cost and duration information to CSP
        """
        self.variables = set(domains)
        self.domains = domains              # domain of csp
        self.constraints = constraints      # constraints of csp
        self.activity = activity            # activity of csp
        self.var_to_const = {var: set() for var in self.variables}
        for con in constraints:
            for var in con.scope:
                self.var_to_const[var].add(con)


class Con_solver(Displayable):
    """Solves a CSP with arc consistency and domain splitting
    """

    def __init__(self, csp, **kwargs):
        """a CSP solver that uses arc consistency
        * csp is the CSP to be solved
        * kwargs is the keyword arguments for Displayable superclass
        """
        self.csp = csp
        super().__init__(**kwargs)  # Or Displayable.__init__(self,**kwargs)

    def make_arc_consistent(self, orig_domains=None, to_do=None):
        """Makes this CSP arc-consistent using generalized arc consistency
        orig_domains is the original domains
        to_do is a set of (variable,constraint) pairs
        returns the reduced domains (an arc-consistent variable:domain dictionary)
        """
        if orig_domains is None:
            orig_domains = self.csp.domains
        if to_do is None:
            to_do = {(var, const) for const in self.csp.constraints
                     for var in const.scope}
        else:
            to_do = to_do.copy()  # use a copy of to_do
        domains = orig_domains.copy()
        self.display(2, "Performing AC with domains", domains)
        while to_do:
            var, const = self.select_arc(to_do)
            self.display(3, "Processing arc (", var, ",", const, ")")
            other_vars = [ov for ov in const.scope if ov != var]
            new_domain = {val for val in domains[var]
                          if self.any_holds(domains, const, {var: val}, other_vars)}
            if new_domain != domains[var]:
                self.display(4, "Arc: (", var, ",", const, ") is inconsistent")
                self.display(3, "Domain pruned", "dom(", var, ") =", new_domain,
                             " due to ", const)
                domains[var] = new_domain
                add_to_do = self.new_to_do(var, const) - to_do
                to_do |= add_to_do  # set union
                self.display(3, "  adding", add_to_do if add_to_do else "nothing", "to to_do.")
            self.display(4, "Arc: (", var, ",", const, ") now consistent")
        self.display(2, "AC done. Reduced domains", domains)
        return domains

    def new_to_do(self, var, const):
        """returns new elements to be added to to_do after assigning
        variable var in constraint const.
        """
        return {(nvar, nconst) for nconst in self.csp.var_to_const[var]
                if nconst != const
                for nvar in nconst.scope
                if nvar != var}

    def select_arc(self, to_do):
        """Selects the arc to be taken from to_do .
        * to_do is a set of arcs, where an arc is a (variable,constraint) pair
        the element selected must be removed from to_do.
        """
        return to_do.pop()

    def any_holds(self, domains, const, env, other_vars, ind=0):
        """returns True if Constraint const holds for an assignment
        that extends env with the variables in other_vars[ind:]
        env is a dictionary
        Warning: this has side effects and changes the elements of env
        """
        if ind == len(other_vars):
            return const.holds(env)
        else:
            var = other_vars[ind]
            for val in domains[var]:
                # env = dict_union(env,{var:val})  # no side effects!
                env[var] = val
                if self.any_holds(domains, const, env, other_vars, ind + 1):
                    return True
            return False

    def solve_one(self, domains=None, to_do=None):
        """return a solution to the current CSP or False if there are no solutions
        to_do is the list of arcs to check
        """
        if domains is None:
            domains = self.csp.domains
        new_domains = self.make_arc_consistent(domains, to_do)
        if any(len(new_domains[var]) == 0 for var in domains):
            return False
        elif all(len(new_domains[var]) == 1 for var in domains):
            self.display(2, "solution:", {var: select(
                new_domains[var]) for var in new_domains})
            return {var: select(new_domains[var]) for var in domains}
        else:
            if len(Cost_dict) > 1:
                var = max(Cost_dict, key=lambda x: Cost_dict[x])
            else:
                var = self.select_var(x for x in self.csp.variables if len(new_domains[x]) > 1)
            ''' try to find the suitable time for soft constraint by computing which time
             has the minimum difference between soft constraint time '''
            if var:
                if 'soft' in activity[var]:
                    min_diff_dict = {}
                    for possible_time in new_domains[var]:
                        x = compute_actual_time(possible_time[0])
                        difference = abs(x-activity[var]['soft'])
                        min_diff_dict[possible_time] = difference
                    aaa = min(min_diff_dict, key=lambda x: min_diff_dict[x])
                    set_var = set()
                    set_var.add(aaa)
                    new_domains[var] = set_var
                    to_do = self.new_to_do(var, None)
                    return self.solve_one(new_domains, to_do)

                else:
                    dom1, dom2 = partition_domain(new_domains[var])
                    self.display(3, "...splitting", var, "into", dom1, "and", dom2)
                    new_doms1 = copy_with_assign(new_domains, var, dom1)
                    new_doms2 = copy_with_assign(new_domains, var, dom2)
                    to_do = self.new_to_do(var, None)
                    self.display(3, " adding", to_do if to_do else "nothing", "to to_do.")
                    return self.solve_one(new_doms1, to_do) or self.solve_one(new_doms2, to_do)

    def select_var(self, iter_vars):
        """return the next variable to split"""
        return select(iter_vars)


def partition_domain(dom):
    """partitions domain dom into two.
    """
    split = len(dom) // 2
    dom1 = set(list(dom)[:split])
    dom2 = dom - dom1  # find out all possible combination for task domains
    return dom1, dom2


def copy_with_assign(domains, var=None, new_domain={True, False}):
    """create a copy of the domains with an assignment var=new_domain
    if var==None then it is just a copy.
    """
    newdoms = domains.copy()
    if var is not None:
        newdoms[var] = new_domain
    return newdoms


def select(iterable):
    """select an element of iterable. Returns None if there is no such element.
    This implementation just picks the first element.
    For many of the uses, which element is selected does not affect correctness,
    but may affect efficiency.
    """
    for e in iterable:
        return e  # returns first element found


# obtain information(domain, constraints, activities) from input file
activity = defaultdict(dict)
Domain_dict = defaultdict(dict)
read_file = []  # to store the information read from file
constraint_list = []  # To store constraint
Cost_dict = {}  # cost information for soft domain
AllTime_List = []  # all of the time from sun 7am to sat 7pm
for day_list in range(0, 7):    # 7 days in a week
    for hour in range(7, 20):    # one day from 7am to 7pm
        AllTime_List.append(hour + day_list * 24)    # put all available time into time_list

# used for calculate the domains and constraints
day_num = {'sun': 0, 'mon': 24, 'tue': 48, 'wed': 72, 'thu': 96, 'fri': 120, 'sat': 144}
day_num_reverse = {'0': 'sun', '1': 'mon', '2': 'tue', '3': 'wed', '4': 'thu', '5': 'fri', '6': 'sat'}


""" binary constraints definition start from here
    define functions to describe every binary constraint
    x[0]/y[0] stands for starting time, x[1]/y[1] stands for ending time """


def before(x, y):
    """
    compare if 2 activities satisfied before binary constraint
    """
    return x[1] <= y[0]


def same_day(x, y):
    """
    compare if 2 activities satisfied same day binary constraint
    """
    if x[1] <= 24 and y[1] <= 24:
        return 1
    if 24 < x[1] <= 24 * 2 and 24 < y[1] <= 24 * 2:
        return 1
    if 24 * 2 < x[1] <= 24 * 3 and 24 * 2 < y[1] <= 24 * 3:
        return 1
    if 24 * 3 < x[1] <= 24 * 4 and 24 * 3 < y[1] <= 24 * 4:
        return 1
    if 24 * 4 < x[1] <= 24 * 5 and 24 * 4 < y[1] <= 24 * 5:
        return 1
    if 24 * 5 < x[1] <= 24 * 6 and 24 * 5 < y[1] <= 24 * 6:
        return 1
    if 24 * 6 < x[1] <= 24 * 7 and 24 * 6 < y[1] <= 24 * 7:
        return 1
    return 0


def after(x, y):
    """
    compare if 2 activities satisfied after binary constraint
    """
    return x[0] >= y[1]


def starts(x, y):
    """
    compare if 2 activities satisfied starts binary constraint
    """
    return x[0] == y[0]


def ends(x, y):
    """
    compare if 2 activities satisfied ends binary constraint
    """
    return x[1] == y[1]


def overlaps(x, y):
    """
    compare if 2 activities satisfied overlaps binary constraint
    """
    return x[0] < y[0] < x[1] < y[1]


def during(x, y):
    """
    compare if 2 activities satisfied during binary constraint
    """
    return x[0] < y[0] and x[1] > y[1]


def equals(x, y):
    """
    compare if 2 activities satisfied equals binary constraint
    """
    return x[0] == y[0] and x[1] == y[1]
# binary constraints definition end here


def on(day):
    """
    convert day in constraint, like 'mon', 'tue' into 24, 48 hours
    """
    x = day_num[day]
    return x


def compute_constraint_time(time):
    """
    convert time in am/pm format into twenty-four hour clock format
    """
    actual_time = []
    if time[-2:] == "pm":
        x = int(time[:-2]) + 12
    else:
        x = int(time[:-2])
    for i in day_num:
        actual_time.append(x + day_num[i])
    return actual_time


# analysis process
# read input
file = sys.stdin.read()
read_file1 = file.split('\n')
for each_line in read_file1:
    read_file.append(each_line.strip("\n").split(" "))
All_Available_Time = AllTime_List.copy()    # copy every hour(time) from sun 7am to sat 7pm to list
for days in read_file:
    if days[0] == 'activity':
        lines_part = days[1::]  # get activity name and duration
        for days in range(1, len(lines_part)):
            activity[lines_part[0]]['duration'] = int(lines_part[days])
            Domain_dict[lines_part[0]] = set(All_Available_Time)    # distribute time to every activities
    elif days[0] == 'domain':
        lines_part = days[1::]
        if lines_part[1] == "around":   # if soft constraint, compute cost
            cost = lines_part[3]
            Cost_dict[lines_part[0]] = int(cost)
            time = lines_part[2]
            if 'pm' in time:
                activity[lines_part[0]]['soft'] = int(time.strip('pm')) + 12
            else:
                activity[lines_part[0]]['soft'] = int(time.strip('am'))

        else:
            Time_not_Qualified = []  # store all time not satisfied constraint
            if Domain_dict[lines_part[0]]:
                hour = list(Domain_dict[lines_part[0]])
                if lines_part[2] in day_num_reverse.values():
                    Start_From_Time = on(lines_part[2])  # compute the activity starts from which day(in hours)
                else:
                    constraint_time = compute_constraint_time(lines_part[2])  # compute the constraint time
                if lines_part[1] == 'on':       # if hard constraint is on
                    for hours in hour:
                        if lines_part[2] == 'sun':  # there is no time before sun, so have take it as a special case
                            if hours > Start_From_Time + 24:
                                Time_not_Qualified.append(hours)
                        else:
                            if hours < Start_From_Time or hours > Start_From_Time + 24:  # other cases beside sun
                                Time_not_Qualified.append(hours)

                elif lines_part[1] == 'before':      # if hard constraint is before
                    for hours in hour:
                        if hours > Start_From_Time:
                            Time_not_Qualified.append(hours)

                elif lines_part[1] == 'after':  # if hard constraint is after
                    for hours in hour:
                        if hours < Start_From_Time:
                            Time_not_Qualified.append(hours)

                elif lines_part[1] == "starts-before":  # if hard constraint is starts before
                    for hours in hour:
                        for days in range(7):
                            if hours <= 19 + 24 * days:
                                if hours > constraint_time[days]:
                                    Time_not_Qualified.append(hours)
                                break
                            else:
                                continue

                elif lines_part[1] == "ends-before":  # if hard constraint is ends before
                    for hours in hour:
                        for days in range(7):
                            if hours <= 19 + 24 * days:
                                if hours + activity[lines_part[0]]['duration'] > constraint_time[days]:
                                    Time_not_Qualified.append(hours)
                                break
                            else:
                                continue

                elif lines_part[1] == "starts-after":   # if hard constraint is starts after
                    for hours in hour:
                        for days in range(7):
                            if hours <= 19 + 24 * days:
                                if hours < constraint_time[days]:
                                    Time_not_Qualified.append(hours)
                                break
                            else:
                                continue

                elif lines_part[1] == "ends-after":     # if hard constraint is ends after
                    for hours in hour:
                        for days in range(7):
                            if hours <= 19 + 24 * days:
                                if hours + activity[lines_part[0]]['duration'] < constraint_time[days]:
                                    Time_not_Qualified.append(hours)
                                break
                            else:
                                continue

                for d in Time_not_Qualified:    # remove all not qualified time
                    if d in hour:
                        hour.remove(d)
                Domain_dict[lines_part[0]] = set(hour)

# if binary constraint, then use constraint object in cspProblem to deal with it
    elif days[0] == 'constraint':
        lines_part = days[1::]
        scope = (lines_part[0], lines_part[2])
        if lines_part[1] == 'before':
            constraint_list.append(Constraint(scope, before))
        elif lines_part[1] == 'same-day':
            constraint_list.append(Constraint(scope, same_day))
        elif lines_part[1] == 'after':
            constraint_list.append(Constraint(scope, after))
        elif lines_part[1] == 'starts':
            constraint_list.append(Constraint(scope, starts))
        elif lines_part[1] == 'ends':
            constraint_list.append(Constraint(scope, ends))
        elif lines_part[1] == 'overlaps':
            constraint_list.append(Constraint(scope, overlaps))
        elif lines_part[1] == 'during':
            constraint_list.append(Constraint(scope, during))
        elif lines_part[1] == 'equals':
            constraint_list.append(Constraint(scope, equals))

# set a new dict to store both starting time and ending time of a activity(in a set),
# like {'lecture': {(82, 85), (81, 84), (79, 82), (80, 83)}, 'tutorial': {(163, 164), (67, 68)}
Domain_dict2 = {}
for key in Domain_dict:
    temp = set()
    for i in Domain_dict[key]:
        temp.add((i, i + activity[key]['duration']))
    Domain_dict2[key] = temp

# use CSP_1 class to build a csp object called weekPlanner
weekPlanner = CSP_1(Domain_dict2, constraint_list, activity)

# use Con_solver.solve_one() to compute the solution
solution = Con_solver(weekPlanner).solve_one()

# if solution exists, then compute the cost
if solution:
    cost = 0
    for days in solution:
        if 'soft' in activity[days]:
            x = solution[days][0]
            while x > 24:
                x -= 24
            cost = abs(x - activity[days]['soft']) * Cost_dict[days] + cost
        else:
            cost = cost + 0
        x = time_transfer(solution[days][0])
        print(days, end=":")
        print(x)
    print(f"cost:{cost}")

# if no solution, print no solution
else:
    print('No solution')
