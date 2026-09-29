Star Kingdom
 
	
Data
 
The island nation of Star Kingdom has a natural gas network of ~43,000 pipes. The network consists of iron, copper, brass and polyurethane pipes. Iron pipes are faulty and can develop costly leaks, whereas polyurethane pipes rarely leak. To avoid repair costs, and to reduce the environmental impact of leaks, the Princess of Star Kingdom has mandated that all iron pipes be replaced with polyurethane pipes within the next 25 years.
 
    Star Kingdom has provided you with a budget of $10,000,000 per year for 25 years (the project window) to complete this replacement work. You must complete this task by specifying a series of work units.
    You are given data about the pipe network (pipes.csv) and eight years of historical data about leaks covering 2019 to 2026 (costs.csv). The project window starts in 2027.
    While your replacement work must result in every iron pipe being replaced, you can choose the size and location of each work unit, and the order in which the work units are conducted. Through these choices, you can:
 
        Prioritize the replacement of pipes that are at risk of incurring costly leaks and replace them before they leak.
        Minimize earthwork costs.
 
Output format
 
Your specification must be a comma separated value file in the following format:
 
    One line per work unit.
    Each work unit is of the form "(x, y)",r (including quotes) where (x, y) is a GPS coordinate (in metres) and r is a radius (in metres), indicating earthwork in a closed circle of radius r centred at (x, y).
    Your specification must be such that all iron pipes are wholly within the circle described by at least one work unit.
    When a work unit is executed, all non-polyurethane pipes wholly within the circle it describes are replaced by polyurethane pipes.
 
Cost of a work unit
Component	Rate
GROUNDBREAKING	$100/work unit
SURFACE – GRASS	$3/m^2
SURFACE – FARM	$7/m^2
SURFACE – SWAMP	$12/m^2
SURFACE – ROAD	$17/m^2
SURFACE – STRUCTURE	$50/m^2
SURFACE – WATER	$32/m^2
LENGTH	$200/m
 
Each work unit has a base cost of $100 which is incurred regardless of its nature. Next a surface rate is formed by summing together all of the rates for all the surface types appearing among all pipes that are wholly within the circle. This surface rate is then multiplied by the area of the circle, and the result of this product is raised to the power 0.85. Added to this is a linear cost of $200 per metre of length for all non-polyurethane pipe wholly within the circle. The cost is then rounded to the nearest cent.
 
cost = 100
     + (surface rate * area)^0.85
     + 200 * (metres of non-polyurethane pipe)
 
Note that the surface rate is not computed based only on the surfaces of the pipes being replaced (instead, it is based on the surfaces of all the pipes in the circle including the ones being replaced and including ones not being replaced). Nor is it determined by examining proportions of surface types appearing among the pipes or the proportions of their lengths. Instead, the surface rate is a sum of indicator functions, which is then multiplied with the entire area of the circle.
Evaluation
 
Your specification is evaluated as follows. The budget starts at $10,000,000. At the beginning of 2027, work units described in your specification are executed in order and their execution costs are subtracted from the budget until a work unit is reached that would bring the budget below zero. These executions are instantaneous. Then, we move to the beginning of 2028, add $10,000,000 to the budget and continue executing work units starting at the one that would have brought the previous year's budget below zero and continuing in the order described in your specification. This process repeats for each year until all work units are completed, or until 25 years have elapsed.
 
    Your specification is rejected if it cannot be completed within the project window, or if any iron pipe remains unreplaced after the execution of all your work units.
    Your surplus is then given by the total budget that remains after there are no more work units left to be executed (i.e., $10,000,000 * 25 - replacement costs).
    Your incurred repair costs are then given by the costs of all the leaks that occur during the 25 year project window.
    Your score is given by your surplus minus your incurred repair cost.
 
Notes
 
    When pipes leak, during the repair, they are replaced by polyurethane pipes. So, each time a leak occurs during the project window, when the work unit containing it is executed, that pipe does not contribute a linear term to the cost of executing the work unit.
    Pipes that leak during the training period have also been replaced by polyurethane (pipes.csv's material indicates the original material).
    If a work unit contains no pipes, or only contains polyurethane pipes (for example because all pipes wholly within its circle have already leaked by the time of execution), it is skipped and does not contribute a replacement cost.
