from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FactExample:
    language: str
    domain: str
    statement: str
    label: int
    pair_id: str


CAPITALS: list[tuple[str, str]] = [
    ("France", "Paris"),
    ("Germany", "Berlin"),
    ("Italy", "Rome"),
    ("Spain", "Madrid"),
    ("Portugal", "Lisbon"),
    ("Netherlands", "Amsterdam"),
    ("Belgium", "Brussels"),
    ("Austria", "Vienna"),
    ("Switzerland", "Bern"),
    ("Poland", "Warsaw"),
    ("Czech Republic", "Prague"),
    ("Hungary", "Budapest"),
    ("Greece", "Athens"),
    ("Norway", "Oslo"),
    ("Sweden", "Stockholm"),
    ("Finland", "Helsinki"),
    ("Denmark", "Copenhagen"),
    ("Ireland", "Dublin"),
    ("United Kingdom", "London"),
    ("Iceland", "Reykjavik"),
    ("Russia", "Moscow"),
    ("Ukraine", "Kyiv"),
    ("Turkey", "Ankara"),
    ("Egypt", "Cairo"),
    ("Morocco", "Rabat"),
    ("Kenya", "Nairobi"),
    ("Ethiopia", "Addis Ababa"),
    ("Nigeria", "Abuja"),
    ("Ghana", "Accra"),
    ("South Africa", "Pretoria"),
    ("India", "New Delhi"),
    ("China", "Beijing"),
    ("Japan", "Tokyo"),
    ("South Korea", "Seoul"),
    ("Thailand", "Bangkok"),
    ("Vietnam", "Hanoi"),
    ("Indonesia", "Jakarta"),
    ("Malaysia", "Kuala Lumpur"),
    ("Singapore", "Singapore"),
    ("Philippines", "Manila"),
    ("Australia", "Canberra"),
    ("New Zealand", "Wellington"),
    ("Canada", "Ottawa"),
    ("United States", "Washington, D.C."),
    ("Mexico", "Mexico City"),
    ("Brazil", "Brasilia"),
    ("Argentina", "Buenos Aires"),
    ("Chile", "Santiago"),
    ("Peru", "Lima"),
    ("Colombia", "Bogota"),
    ("Venezuela", "Caracas"),
    ("Uruguay", "Montevideo"),
    ("Paraguay", "Asuncion"),
    ("Bolivia", "Sucre"),
    ("Ecuador", "Quito"),
    ("Saudi Arabia", "Riyadh"),
    ("Iran", "Tehran"),
    ("Iraq", "Baghdad"),
    ("Israel", "Jerusalem"),
    ("Jordan", "Amman"),
    ("Lebanon", "Beirut"),
    ("Pakistan", "Islamabad"),
    ("Afghanistan", "Kabul"),
    ("Kazakhstan", "Astana"),
    ("Mongolia", "Ulaanbaatar"),
    ("Nepal", "Kathmandu"),
    ("Bangladesh", "Dhaka"),
    ("Sri Lanka", "Sri Jayawardenepura Kotte"),
    ("Myanmar", "Naypyidaw"),
    ("Cambodia", "Phnom Penh"),
    ("Laos", "Vientiane"),
    ("Cuba", "Havana"),
    ("Jamaica", "Kingston"),
    ("Dominican Republic", "Santo Domingo"),
    ("Costa Rica", "San Jose"),
    ("Panama", "Panama City"),
]


COUNTRY_CONTINENTS: list[tuple[str, str]] = [
    ("France", "Europe"),
    ("Germany", "Europe"),
    ("Italy", "Europe"),
    ("Spain", "Europe"),
    ("Norway", "Europe"),
    ("Sweden", "Europe"),
    ("Poland", "Europe"),
    ("Greece", "Europe"),
    ("Egypt", "Africa"),
    ("Morocco", "Africa"),
    ("Kenya", "Africa"),
    ("Ethiopia", "Africa"),
    ("Nigeria", "Africa"),
    ("Ghana", "Africa"),
    ("South Africa", "Africa"),
    ("Algeria", "Africa"),
    ("India", "Asia"),
    ("China", "Asia"),
    ("Japan", "Asia"),
    ("South Korea", "Asia"),
    ("Thailand", "Asia"),
    ("Vietnam", "Asia"),
    ("Indonesia", "Asia"),
    ("Saudi Arabia", "Asia"),
    ("Australia", "Australia"),
    ("New Zealand", "Australia"),
    ("Fiji", "Australia"),
    ("Papua New Guinea", "Australia"),
    ("Canada", "North America"),
    ("United States", "North America"),
    ("Mexico", "North America"),
    ("Cuba", "North America"),
    ("Jamaica", "North America"),
    ("Panama", "North America"),
    ("Brazil", "South America"),
    ("Argentina", "South America"),
    ("Chile", "South America"),
    ("Peru", "South America"),
    ("Colombia", "South America"),
    ("Uruguay", "South America"),
    ("Paraguay", "South America"),
    ("Bolivia", "South America"),
    ("Ecuador", "South America"),
]


ELEMENT_SYMBOLS: list[tuple[str, str]] = [
    ("hydrogen", "H"),
    ("helium", "He"),
    ("lithium", "Li"),
    ("beryllium", "Be"),
    ("boron", "B"),
    ("carbon", "C"),
    ("nitrogen", "N"),
    ("oxygen", "O"),
    ("fluorine", "F"),
    ("neon", "Ne"),
    ("sodium", "Na"),
    ("magnesium", "Mg"),
    ("aluminum", "Al"),
    ("silicon", "Si"),
    ("phosphorus", "P"),
    ("sulfur", "S"),
    ("chlorine", "Cl"),
    ("argon", "Ar"),
    ("potassium", "K"),
    ("calcium", "Ca"),
    ("iron", "Fe"),
    ("copper", "Cu"),
    ("zinc", "Zn"),
    ("silver", "Ag"),
    ("gold", "Au"),
    ("mercury", "Hg"),
    ("lead", "Pb"),
    ("tin", "Sn"),
    ("iodine", "I"),
    ("uranium", "U"),
    ("nickel", "Ni"),
    ("cobalt", "Co"),
    ("chromium", "Cr"),
    ("manganese", "Mn"),
    ("bromine", "Br"),
    ("krypton", "Kr"),
    ("xenon", "Xe"),
    ("barium", "Ba"),
    ("platinum", "Pt"),
    ("tungsten", "W"),
]


BOOK_AUTHORS: list[tuple[str, str]] = [
    ("1984", "George Orwell"),
    ("Animal Farm", "George Orwell"),
    ("Pride and Prejudice", "Jane Austen"),
    ("Emma", "Jane Austen"),
    ("Jane Eyre", "Charlotte Bronte"),
    ("Wuthering Heights", "Emily Bronte"),
    ("Moby-Dick", "Herman Melville"),
    ("The Great Gatsby", "F. Scott Fitzgerald"),
    ("The Catcher in the Rye", "J. D. Salinger"),
    ("To Kill a Mockingbird", "Harper Lee"),
    ("The Hobbit", "J. R. R. Tolkien"),
    ("The Lord of the Rings", "J. R. R. Tolkien"),
    ("Harry Potter and the Philosopher's Stone", "J. K. Rowling"),
    ("The Old Man and the Sea", "Ernest Hemingway"),
    ("A Farewell to Arms", "Ernest Hemingway"),
    ("The Adventures of Tom Sawyer", "Mark Twain"),
    ("Adventures of Huckleberry Finn", "Mark Twain"),
    ("Great Expectations", "Charles Dickens"),
    ("Oliver Twist", "Charles Dickens"),
    ("War and Peace", "Leo Tolstoy"),
    ("Anna Karenina", "Leo Tolstoy"),
    ("Crime and Punishment", "Fyodor Dostoevsky"),
    ("The Brothers Karamazov", "Fyodor Dostoevsky"),
    ("One Hundred Years of Solitude", "Gabriel Garcia Marquez"),
    ("The Stranger", "Albert Camus"),
    ("Frankenstein", "Mary Shelley"),
    ("Dracula", "Bram Stoker"),
    ("The Odyssey", "Homer"),
    ("The Iliad", "Homer"),
    ("The Divine Comedy", "Dante Alighieri"),
]


LANDMARK_COUNTRIES: list[tuple[str, str]] = [
    ("the Eiffel Tower", "France"),
    ("the Colosseum", "Italy"),
    ("the Leaning Tower of Pisa", "Italy"),
    ("the Acropolis", "Greece"),
    ("the Great Wall", "China"),
    ("the Taj Mahal", "India"),
    ("Machu Picchu", "Peru"),
    ("Christ the Redeemer", "Brazil"),
    ("the Pyramids of Giza", "Egypt"),
    ("Petra", "Jordan"),
    ("Stonehenge", "United Kingdom"),
    ("the Statue of Liberty", "United States"),
    ("Mount Rushmore", "United States"),
    ("the Sydney Opera House", "Australia"),
    ("Angkor Wat", "Cambodia"),
    ("the Burj Khalifa", "United Arab Emirates"),
    ("the Kremlin", "Russia"),
    ("Neuschwanstein Castle", "Germany"),
    ("the Louvre Museum", "France"),
    ("the Sagrada Familia", "Spain"),
    ("Chichen Itza", "Mexico"),
    ("the Palace of Versailles", "France"),
    ("the Forbidden City", "China"),
    ("the Blue Mosque", "Turkey"),
    ("the Tower of London", "United Kingdom"),
    ("the Grand Palace", "Thailand"),
    ("Mount Fuji", "Japan"),
    ("Table Mountain", "South Africa"),
    ("the CN Tower", "Canada"),
    ("the Panama Canal", "Panama"),
]


SCIENCE_PAIRS: list[tuple[str, str, str]] = [
    ("Water freezes at 0 degrees Celsius at standard pressure.", "Water freezes at 100 degrees Celsius at standard pressure.", "water_freezing"),
    ("Water boils at 100 degrees Celsius at standard pressure.", "Water boils at 0 degrees Celsius at standard pressure.", "water_boiling"),
    ("The Earth orbits the Sun.", "The Sun orbits the Earth.", "earth_orbit"),
    ("The Moon orbits the Earth.", "The Earth orbits the Moon.", "moon_orbit"),
    ("Humans need oxygen to survive.", "Humans need helium to survive.", "human_oxygen"),
    ("Plants use photosynthesis to convert light energy into chemical energy.", "Plants use photosynthesis to convert sound into gravity.", "photosynthesis"),
    ("DNA carries genetic information in living organisms.", "DNA carries electrical power in household wiring.", "dna"),
    ("The heart pumps blood through the body.", "The heart pumps air into the lungs.", "heart"),
    ("The brain is part of the nervous system.", "The brain is part of the digestive system.", "brain"),
    ("Electrons have a negative electric charge.", "Electrons have a positive electric charge.", "electron_charge"),
    ("Protons have a positive electric charge.", "Protons have a negative electric charge.", "proton_charge"),
    ("A triangle has three sides.", "A triangle has four sides.", "triangle"),
    ("A square has four equal sides.", "A square has three equal sides.", "square"),
    ("Sound travels through air as waves.", "Sound travels through air as solid cubes.", "sound"),
    ("Gravity attracts masses toward each other.", "Gravity makes masses repel each other.", "gravity"),
    ("The Pacific Ocean is the largest ocean on Earth.", "The Arctic Ocean is the largest ocean on Earth.", "largest_ocean"),
    ("Mars is often called the Red Planet.", "Venus is often called the Red Planet.", "red_planet"),
    ("Jupiter is the largest planet in the Solar System.", "Mercury is the largest planet in the Solar System.", "largest_planet"),
    ("Mercury is the closest planet to the Sun.", "Neptune is the closest planet to the Sun.", "closest_planet"),
    ("Saturn has prominent rings.", "Mercury has prominent rings.", "saturn_rings"),
    ("The human skeleton contains bones.", "The human skeleton contains feathers.", "skeleton"),
    ("Mammals are warm-blooded vertebrates.", "Mammals are cold-blooded plants.", "mammals"),
    ("Birds usually have feathers.", "Birds usually have scales instead of feathers.", "birds"),
    ("Fish live in water and breathe using gills.", "Fish live in fire and breathe using leaves.", "fish"),
    ("Carbon dioxide contains carbon and oxygen.", "Carbon dioxide contains gold and helium.", "carbon_dioxide"),
]


MATH_PAIRS: list[tuple[str, str, str]] = [
    ("Two plus two equals four.", "Two plus two equals five.", "add_2_2"),
    ("Three plus five equals eight.", "Three plus five equals nine.", "add_3_5"),
    ("Seven minus four equals three.", "Seven minus four equals two.", "sub_7_4"),
    ("Nine minus six equals three.", "Nine minus six equals four.", "sub_9_6"),
    ("Six times seven equals forty-two.", "Six times seven equals forty-one.", "mul_6_7"),
    ("Eight times eight equals sixty-four.", "Eight times eight equals sixty-three.", "mul_8_8"),
    ("Twelve divided by three equals four.", "Twelve divided by three equals five.", "div_12_3"),
    ("Fifteen divided by five equals three.", "Fifteen divided by five equals four.", "div_15_5"),
    ("The square root of nine is three.", "The square root of nine is four.", "sqrt_9"),
    ("The square root of sixteen is four.", "The square root of sixteen is five.", "sqrt_16"),
    ("Ten is greater than seven.", "Ten is less than seven.", "compare_10_7"),
    ("Five is less than eleven.", "Five is greater than eleven.", "compare_5_11"),
    ("A right angle measures ninety degrees.", "A right angle measures forty-five degrees.", "right_angle"),
    ("A full circle has 360 degrees.", "A full circle has 180 degrees.", "circle_degrees"),
    ("A prime number has exactly two positive divisors.", "A prime number has exactly three positive divisors.", "prime"),
    ("The number two is prime.", "The number two is composite.", "two_prime"),
    ("The number nine is composite.", "The number nine is prime.", "nine_composite"),
    ("Zero is an even number.", "Zero is an odd number.", "zero_even"),
    ("One kilometer equals 1000 meters.", "One kilometer equals 100 meters.", "kilometer"),
    ("One hour contains 60 minutes.", "One hour contains 100 minutes.", "hour"),
]


def _slug(text: str) -> str:
    return (
        text.lower()
        .replace(", d.c.", " dc")
        .replace(".", "")
        .replace("'", "")
        .replace("-", " ")
        .replace(" ", "_")
    )


def _rotated(values: list[str], index: int, offset: int) -> str:
    return values[(index + offset) % len(values)]


def _rotated_not_equal(values: list[str], current: str, index: int, offset: int) -> str:
    for extra_offset in range(len(values)):
        candidate = values[(index + offset + extra_offset) % len(values)]
        if candidate != current:
            return candidate
    raise ValueError("Could not find a rotated value different from the current value.")


def _add_pair(
    rows: list[dict[str, object]],
    *,
    domain: str,
    pair_id: str,
    true_statement: str,
    false_statement: str,
    language: str = "en",
) -> None:
    rows.append(
        {
            "language": language,
            "domain": domain,
            "pair_id": pair_id,
            "statement": true_statement,
            "label": 1,
        }
    )
    rows.append(
        {
            "language": language,
            "domain": domain,
            "pair_id": pair_id,
            "statement": false_statement,
            "label": 0,
        }
    )


def build_fact_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    capital_values = [capital for _, capital in CAPITALS]
    for i, (country, capital) in enumerate(CAPITALS):
        false_capital = _rotated(capital_values, i, 17)
        _add_pair(
            rows,
            domain="capital",
            pair_id=f"capital_{_slug(country)}",
            true_statement=f"The capital of {country} is {capital}.",
            false_statement=f"The capital of {country} is {false_capital}.",
        )

    continents = ["Africa", "Asia", "Europe", "North America", "South America", "Australia"]
    for i, (country, continent) in enumerate(COUNTRY_CONTINENTS):
        false_continent = continents[(continents.index(continent) + 2 + i % 3) % len(continents)]
        _add_pair(
            rows,
            domain="continent",
            pair_id=f"continent_{_slug(country)}",
            true_statement=f"{country} is in {continent}.",
            false_statement=f"{country} is in {false_continent}.",
        )

    symbols = [symbol for _, symbol in ELEMENT_SYMBOLS]
    for i, (element, symbol) in enumerate(ELEMENT_SYMBOLS):
        false_symbol = _rotated(symbols, i, 11)
        _add_pair(
            rows,
            domain="element_symbol",
            pair_id=f"element_{_slug(element)}",
            true_statement=f"The chemical symbol for {element} is {symbol}.",
            false_statement=f"The chemical symbol for {element} is {false_symbol}.",
        )

    authors = [author for _, author in BOOK_AUTHORS]
    for i, (book, author) in enumerate(BOOK_AUTHORS):
        false_author = _rotated(authors, i, 7)
        _add_pair(
            rows,
            domain="book_author",
            pair_id=f"book_{_slug(book)}",
            true_statement=f"{author} wrote {book}.",
            false_statement=f"{false_author} wrote {book}.",
        )

    landmark_countries = sorted(set(country for _, country in LANDMARK_COUNTRIES))
    for i, (landmark, country) in enumerate(LANDMARK_COUNTRIES):
        false_country = _rotated_not_equal(landmark_countries, country, i, 9)
        _add_pair(
            rows,
            domain="landmark_country",
            pair_id=f"landmark_{_slug(landmark)}",
            true_statement=f"{landmark} is in {country}.",
            false_statement=f"{landmark} is in {false_country}.",
        )

    for true_statement, false_statement, pair_id in SCIENCE_PAIRS:
        _add_pair(
            rows,
            domain="science",
            pair_id=f"science_{pair_id}",
            true_statement=true_statement,
            false_statement=false_statement,
        )

    for true_statement, false_statement, pair_id in MATH_PAIRS:
        _add_pair(
            rows,
            domain="math",
            pair_id=f"math_{pair_id}",
            true_statement=true_statement,
            false_statement=false_statement,
        )

    return rows


def save_default_dataset(path: str | Path = "data/facts.csv") -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = build_fact_rows()
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["language", "domain", "pair_id", "statement", "label"])
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def load_dataset(path: str | Path):
    import pandas as pd

    data = pd.read_csv(path)
    expected = {"language", "domain", "pair_id", "statement", "label"}
    missing = expected.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return data


def filter_dataset(data, language: str | None = None, domain: str | None = None):
    filtered = data
    if language:
        filtered = filtered[filtered["language"] == language]
    if domain:
        domains = {item.strip() for item in domain.split(",")}
        filtered = filtered[filtered["domain"].isin(domains)]
    if filtered.empty:
        raise ValueError("Dataset filter removed all rows.")
    return filtered.reset_index(drop=True)
