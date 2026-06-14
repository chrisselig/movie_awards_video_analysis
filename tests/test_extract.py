"""Tests for extract_data.py parsing functions."""
from extract_data import parse_year, parse_category, filter_movie_awards


def test_parse_year_from_of_pattern():
    assert parse_year("Top 30 Best Movies of 2024, Part 3! | 11th Justin & Tyler Movie Awards") == 2024


def test_parse_year_from_standalone():
    assert parse_year("The 2017 BridgewatersFinest Movie Awards -- Top 30 Best Movies") == 2017


def test_parse_year_returns_none_for_no_year():
    assert parse_year("Some Random Title Without Year") is None


def test_parse_year_mid_year():
    assert parse_year("PART 1 | The 2023 Justin & Tyler MID-YEAR Movie Awards") == 2023


def test_parse_category_opening_ceremony():
    assert parse_category("Opening Ceremony | 12th Justin & Tyler Movie Awards") == "Opening Ceremony"


def test_parse_category_worst_movies():
    assert parse_category("Top 5 Worst Movies of 2025 | 12th Justin & Tyler Movie Awards") == "Worst Movies"


def test_parse_category_best_movies():
    assert parse_category("Best Movies of 2025 PART THREE | 12th Justin & Tyler Movie Awards") == "Best Movies"


def test_parse_category_top_ten():
    assert parse_category("TOP TEN Best Movies of 2020! | The 2020 Justin & Tyler Movie Awards") == "Best Movies"


def test_parse_category_fans_choice():
    assert parse_category("FAN'S CHOICE! Best Movies of 2025 | 12th Justin & Tyler Movie Awards") == "Fan's Choice"


def test_parse_category_best_actresses():
    assert parse_category("Best Actresses of 2025 | 12th Justin & Tyler Movie Awards") == "Best Actresses"


def test_parse_category_best_actors():
    assert parse_category("Best Actors of 2025 | 12th Justin & Tyler Movie Awards") == "Best Actors"


def test_parse_category_directors():
    assert parse_category("Best Directors of 2025 | 12th Justin & Tyler Movie Awards") == "Best Directors"


def test_parse_category_miscellaneous():
    assert parse_category("Miscellaneous Movie Awards | 12th Justin & Tyler Movie Awards") == "Miscellaneous"


def test_parse_category_mid_year():
    assert parse_category("PART 2 | The 2025 Justin & Tyler MID-YEAR Movie Awards") == "Mid-Year Awards"


def test_parse_category_retrospective():
    assert parse_category("The Justin & Tyler Movie Awards - A Retrospective (2023 Edition)") == "Retrospective"


def test_filter_movie_awards_includes_matching():
    videos = [
        {"title": "Best Movies of 2025 | 12th Justin & Tyler Movie Awards", "video_id": "a", "duration": 100},
        {"title": "Week 21 | Justin's 2025 CFL Football Picks Show", "video_id": "b", "duration": 100},
        {"title": "The 2017 BridgewatersFinest Movie Awards -- Opening Ceremony", "video_id": "c", "duration": 100},
    ]
    result = filter_movie_awards(videos)
    assert len(result) == 2
    assert result[0]["video_id"] == "a"
    assert result[1]["video_id"] == "c"


def test_filter_movie_awards_case_insensitive():
    videos = [{"title": "MOVIE AWARDS thing", "video_id": "x", "duration": 0}]
    assert len(filter_movie_awards(videos)) == 1


def test_filter_movie_awards_empty_list():
    assert filter_movie_awards([]) == []
