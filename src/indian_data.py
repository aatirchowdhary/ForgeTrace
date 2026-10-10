"""Fictional Indian identity data. Everything here is randomly generated; nothing refers to a real person."""
from __future__ import annotations

import datetime
import random
import string

MALE_FIRST = [
    "Aarav", "Vivaan", "Aditya", "Arjun", "Rohan", "Rahul", "Amit", "Vikram", "Suresh", "Ramesh",
    "Mahesh", "Rajesh", "Sanjay", "Anil", "Sunil", "Manoj", "Deepak", "Pankaj", "Nitin", "Ashok",
    "Vijay", "Kiran", "Prakash", "Harish", "Gopal", "Mohan", "Krishna", "Sachin", "Naveen", "Karthik",
    "Venkatesh", "Srinivas", "Ravi", "Arun", "Ajay", "Abhishek", "Yash", "Ishaan", "Kabir", "Dhruv",
    "Siddharth", "Varun", "Tarun", "Manish", "Girish", "Dinesh", "Jagdish", "Lokesh", "Mukesh", "Nikhil",
    "Pranav", "Rakesh", "Satish", "Tushar", "Uday", "Vinod", "Yogesh", "Zubin", "Imran", "Faisal",
]
FEMALE_FIRST = [
    "Aadhya", "Ananya", "Diya", "Isha", "Kavya", "Meera", "Neha", "Priya", "Pooja", "Riya",
    "Sneha", "Sunita", "Anita", "Geeta", "Radha", "Sita", "Lakshmi", "Sarita", "Kavita", "Rekha",
    "Nisha", "Divya", "Shreya", "Tanvi", "Aishwarya", "Swati", "Pallavi", "Anjali", "Ritu", "Shweta",
    "Jyoti", "Seema", "Rani", "Usha", "Vandana", "Manisha", "Madhuri", "Komal", "Bhavna", "Charu",
    "Deepa", "Esha", "Gauri", "Harini", "Indira", "Jaya", "Lata", "Mala", "Nandini", "Padma",
    "Rashmi", "Savita", "Tara", "Uma", "Vidya", "Yamini", "Zoya", "Fatima", "Ayesha", "Bindu",
]
FIRST_NAMES = MALE_FIRST + FEMALE_FIRST

SURNAMES = [
    "Sharma", "Verma", "Gupta", "Singh", "Kumar", "Patel", "Shah", "Mehta", "Joshi", "Desai",
    "Iyer", "Iyengar", "Nair", "Menon", "Pillai", "Reddy", "Rao", "Naidu", "Chowdhury", "Banerjee",
    "Mukherjee", "Chatterjee", "Ghosh", "Das", "Bose", "Sen", "Dutta", "Roy", "Sinha", "Mishra",
    "Pandey", "Tiwari", "Dubey", "Yadav", "Chauhan", "Rathore", "Thakur", "Kapoor", "Khanna", "Malhotra",
    "Chopra", "Bhatia", "Arora", "Sethi", "Grover", "Gill", "Sandhu", "Dhillon", "Bajwa", "Khan",
    "Ansari", "Qureshi", "Siddiqui", "Sheikh", "Mirza", "Pathan", "Kulkarni", "Deshpande", "Patil", "Jadhav",
    "Shinde", "Pawar", "More", "Gaikwad", "Bhosale", "Kadam", "Naik", "Fernandes", "Pereira", "Gomes",
    "Lobo", "Rodrigues", "Menezes", "Kamath", "Shetty", "Hegde", "Bhat", "Acharya", "Gowda", "Murthy",
    "Subramanian", "Krishnan", "Raman", "Chandra", "Prasad", "Jha", "Thakkar", "Trivedi", "Pandya", "Parekh",
    "Modi", "Soni", "Agarwal", "Bansal", "Goyal", "Jain", "Mittal", "Saxena", "Srivastava", "Kashyap",
    "Bhardwaj", "Tripathi", "Shukla", "Mahajan", "Bhatt", "Rawat", "Negi", "Bisht",
]

STATES = {
    "Andhra Pradesh": ["Visakhapatnam", "Vijayawada", "Guntur"],
    "Arunachal Pradesh": ["Itanagar", "Naharlagun", "Tawang"],
    "Assam": ["Guwahati", "Dibrugarh", "Silchar"],
    "Bihar": ["Patna", "Gaya", "Muzaffarpur"],
    "Chhattisgarh": ["Raipur", "Bhilai", "Bilaspur"],
    "Goa": ["Panaji", "Margao", "Vasco da Gama"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara"],
    "Haryana": ["Gurugram", "Faridabad", "Panipat"],
    "Himachal Pradesh": ["Shimla", "Dharamshala", "Mandi"],
    "Jharkhand": ["Ranchi", "Jamshedpur", "Dhanbad"],
    "Karnataka": ["Bengaluru", "Mysuru", "Hubballi"],
    "Kerala": ["Kochi", "Thiruvananthapuram", "Kozhikode"],
    "Madhya Pradesh": ["Bhopal", "Indore", "Gwalior"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur"],
    "Manipur": ["Imphal", "Thoubal", "Churachandpur"],
    "Meghalaya": ["Shillong", "Tura", "Jowai"],
    "Mizoram": ["Aizawl", "Lunglei", "Champhai"],
    "Nagaland": ["Kohima", "Dimapur", "Mokokchung"],
    "Odisha": ["Bhubaneswar", "Cuttack", "Rourkela"],
    "Punjab": ["Ludhiana", "Amritsar", "Jalandhar"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur"],
    "Sikkim": ["Gangtok", "Namchi", "Gyalshing"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai"],
    "Telangana": ["Hyderabad", "Warangal", "Nizamabad"],
    "Tripura": ["Agartala", "Udaipur", "Dharmanagar"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Varanasi"],
    "Uttarakhand": ["Dehradun", "Haridwar", "Nainital"],
    "West Bengal": ["Kolkata", "Howrah", "Siliguri"],
}

assert len(FIRST_NAMES) >= 100 and len(SURNAMES) >= 100 and len(STATES) == 28

# ---------------------------------------------------------------- Verhoeff
_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]
_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_checksum(num: str) -> int:
    """Check digit to APPEND to `num` so that the result is Verhoeff-valid."""
    c = 0
    for i, ch in enumerate(reversed(num)):
        c = _D[c][_P[(i + 1) % 8][int(ch)]]
    return _INV[c]


def verhoeff_valid(num: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(num)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0


def generate_aadhaar(rng: random.Random, spaced: bool = True) -> str:
    body = str(rng.randint(2, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(10))
    num = body + str(verhoeff_checksum(body))
    return f"{num[:4]} {num[4:8]} {num[8:]}" if spaced else num


def generate_pan(rng: random.Random, surname: str) -> str:
    L = string.ascii_uppercase
    return "".join(rng.choice(L) for _ in range(3)) + "P" + surname[0].upper() + \
        "".join(str(rng.randint(0, 9)) for _ in range(4)) + rng.choice(L)


# ---------------------------------------------------------------- person
def random_dob(rng: random.Random, min_year: int = 1960, max_year: int = 2005) -> str:
    start, end = datetime.date(min_year, 1, 1), datetime.date(max_year, 12, 31)
    d = start + datetime.timedelta(days=rng.randint(0, (end - start).days))
    return d.strftime("%d/%m/%Y")


def random_pin(rng: random.Random) -> str:
    return str(rng.randint(1, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(5))


def random_full_name(rng: random.Random, gender: str | None = None, surname: str | None = None) -> str:
    pool = MALE_FIRST if gender == "Male" else FEMALE_FIRST if gender == "Female" else FIRST_NAMES
    return f"{rng.choice(pool)} {surname or rng.choice(SURNAMES)}"


def make_person(rng: random.Random, min_year: int = 1960, max_year: int = 2005) -> dict:
    gender = rng.choice(["Male", "Female"])
    surname = rng.choice(SURNAMES)
    first = rng.choice(MALE_FIRST if gender == "Male" else FEMALE_FIRST)
    state = rng.choice(list(STATES))
    return {
        "first": first,
        "surname": surname,
        "name": f"{first} {surname}",
        "gender": gender,
        "father": random_full_name(rng, "Male", surname),
        "dob": random_dob(rng, min_year, max_year),
        "pin": random_pin(rng),
        "state": state,
        "city": rng.choice(STATES[state]),
    }
