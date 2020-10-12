# EasyRider
## Empowering cyclists to discover their favourite ride
### Insight - NY.DS 20C

## Purpose
Road biking is a hugely popular activity (e.g. [Ulster County Non-Motorised Transportation, 2008](https://ulstercountyny.gov/sites/default/files/nonmotorizedtranplan_finalplan.pdf)).  It is also very important economically, with $83 billion spent annually on bicycle tourism in the US ([2017 National Recreation Economy Report](http://oia.outdoorindustry.org/OIA-receconomy)).

However, it can often be hard to find a new ride - with advice ranging from 'make a local friend' to 'turn down a random side road' ([Cycling Weekly](https://www.cyclingweekly.com/fitness/training/seven-ways-to-find-great-new-places-to-ride-164073)).  EasyRider fills this niche, by allowing users to submit their preferences (e.g. start location, distance, amount of climbing, popularity, picturesqueness) and returning the nearest neighbour rides from a database of cleaned rides scraped from Ride With GPS.

The app is deployed to Heroku at [insight-easy-rider.herokuapp.com](insight-easy-rider.herokuapp.com), via the GitHub repo [EasyRider-Deployment](https://github.com/EHopper/EasyRider-Deployment).

## Organisation
This project can be followed through Jupyter Notebooks, and uses a number of custom functions.

### 01.  Get and Clean Data
Get the data from Ride With GPS - first by searching for ride IDs within the vicinity of an input location (here, 25 miles of Shokan, NY), then downloading higher resolution data for each of these rides.  

Cleaning steps include unit conversion and removal of rides that are either uninterestingly short, or do not seem humanly possible, e.g. the GPS unit was accidentally left on after the bicycle was loaded back onto the car - or maybe some people with true legs of steel can cycle at 70 mph all the way to Boston :)

### 02. Clean up with Clustering
Here, we tidy the dataset using clustering

1. DBSCAN to identify recording errors
People often have errors in using their GPS units, e.g. pausing for lunch and forgetting to turn the unit back on straight away.  Machine error is also possible, with large gaps between recorded points.  In either case, the recorded ride is not a good recommendation, as the ride is not continuous.  Here, we use DBSCAN to make sure that all of the rides (or longest ride segments) we are considering are relatively continuous, with points no further apart than some distance threshold (here, roughly 1 km).

2. CLIQUE clustering to get a road backbone
The lat/lon breadcrumbs are recorded every 1-2 seconds, with some location error, for every ride.  In the original database of 20,000 rides, I have 35 million unique locations.  I use CLIQUE clustering to create a consistent road backbone of cluster locations.  CLIQUE clustering allows me to set the granularity of the cluster locations, and generates a backbone of approximately evenly spaced cluster locations wherever the data is above some threshold (here, a single ride).  It does not care about putting more cluster locations where data is more densely recorded.  Therefore, it generates a representative road backbone.

With this consistent road backbone, I can find and remove duplicates; calculate features that are location specific, e.g. popularity, picturesqueness; reduce latency when calculating distance from a user's input start location to every route in the database; store the data much more efficiently (4 GB down to 2 MB).

3. DBSCAN to identify duplicates
Using a coarse grid for CLIQUE clustering, we can create a one-hot encoding for location, where every row is a different ride and every feature is a Boolean for whether that ride passes through a unique grid point.  We then want to identify clusters in this space.  Agglomerative clustering would be ideal, but is very slow to run given the large, sparse feature space. DBSCAN (the sklearn implementation) can be used with a sparse matrix as input, making it much faster.  Here, we set the maximum distance between points to be < 1, i.e. all of the grid points in a cluster are overlapping.  Given the coarseness of the chosen CLIQUE clustering grid, this proves capable of identifying near-duplicates.
