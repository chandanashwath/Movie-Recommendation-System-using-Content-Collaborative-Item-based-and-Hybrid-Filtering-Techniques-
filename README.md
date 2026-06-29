The movies.csv file contains a curated catalog of 50 fictional Hollywood‑style movies with rich
metadata suitable for content‑based recommendation. Each record includes:

-movie_id: Unique numeric identifier
- title: Movie title
-year: Release year between 1990 and 2024
- genres: Pipe‑separated list of genres (e.g., “Comedy”, “Action”, “Thriller”)
- director: Single director name
- actors: Multiple high‑profile actors concatenated in one field
- runtime: Approximate duration in minutes
- avg_rating: Aggregate mean rating on a 1–10 scale
- budget_millions: Approximate budget in millions of dollars


The users.csv file represents 100 synthetic users, including demographic and preference‑oriented
attributes. For each user, the following fields are provided:

-user_id: Unique numeric identifier
-age: Age spanning approximately 18–70 years
-gender: {M, F, Other}
-favorite_genres: Pipe‑separated list of preferred genres, such as “Action”, “Romance”,“Documentary”, “Sci‑Fi”
-join_date: Date when the user joined the platform between 2023 and 2025 


The ratings.csv file consists of approximately 1,000 user–movie interaction records, representing
explicit ratings between 1.5 and 5.0, with associated timestamps from late 2024 to 2025. Each row
contains:
-user_id
-movie_id
-rating: Floating‑point rating
-timestamp: Rating Date

