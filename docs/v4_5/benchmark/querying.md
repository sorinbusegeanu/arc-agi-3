## v4.5 Benchmark Querying

Supported queries:

- list all games
- list active benchmark games
- show latest run for a game
- show best result for a game
- show history for a game
- show history for a level in a game
- compare two runs
- show leaderboard by solved levels and by steps

The query layer reads persisted benchmark data only. It does not re-run evaluation logic when answering queries.
