"""Offline map matching and along-road tracking.

Three pieces, each usable on its own:

* `extent`    -- the geographic bounding box the offline map must cover,
                 derived from the test sessions rather than hard-coded.
* `osrm`      -- a client for a LOCAL osrm-routed process. Binds 127.0.0.1
                 only; there is no network at inference.
* `graph`     -- the drivable road graph (polylines + junction adjacency)
                 read from the same cropped extract OSRM was built from.
                 This is the DEPLOYABLE path: matching runs in-process
                 against road_graph.npz with no server. OSRM remains
                 available as a comparison backend via MapMatchConfig
                 (backend="osrm").
* `predictor` -- `MapMatchedPredictor`, a harness predictor that turns a
                 displacement model into an along-road tracker.
"""
