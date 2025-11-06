import pandas as pd
import numpy as np
import random
from datetime import datetime

def generate_crime_dataset():
    """Generate realistic synthetic crime dataset for Tamil Nadu with logical correlations"""
    
    # Tamil Nadu districts with their taluks
    district_taluks = {
        'Chennai': ['Chennai North', 'Chennai South', 'Chennai Central', 'Ambattur', 'Sholinganallur'],
        'Coimbatore': ['Coimbatore North', 'Coimbatore South', 'Pollachi', 'Mettupalayam', 'Sulur'],
        'Madurai': ['Madurai North', 'Madurai South', 'Melur', 'Usilampatti', 'Thirumangalam'],
        'Tiruchirappalli': ['Tiruchirappalli', 'Srirangam', 'Lalgudi', 'Musiri', 'Thuraiyur'],
        'Salem': ['Salem', 'Mettur', 'Omalur', 'Sankagiri', 'Vazhapadi'],
        'Tirunelveli': ['Tirunelveli', 'Ambasamudram', 'Nanguneri', 'Radhapuram', 'Palayamkottai'],
        'Erode': ['Erode', 'Gobichettipalayam', 'Anthiyur', 'Bhavani', 'Modakurichi'],
        'Vellore': ['Vellore', 'Arcot', 'Gudiyatham', 'Katpadi', 'Pernambut'],
        'Thoothukudi': ['Thoothukudi', 'Kovilpatti', 'Ottapidaram', 'Vilathikulam', 'Srivaikuntam'],
        'Thanjavur': ['Thanjavur', 'Kumbakonam', 'Papanasam', 'Pattukottai', 'Thiruvidaimarudur'],
        'Dindigul': ['Dindigul', 'Kodaikanal', 'Natham', 'Nilakottai', 'Palani'],
        'Cuddalore': ['Cuddalore', 'Chidambaram', 'Kattumannarkoil', 'Panruti', 'Vridhachalam'],
        'Kanchipuram': ['Kanchipuram', 'Chengalpattu', 'Madurantakam', 'Sriperumbudur', 'Uthiramerur'],
        'Villupuram': ['Villupuram', 'Gingee', 'Kallakurichi', 'Sankarapuram', 'Tindivanam'],
        'Sivaganga': ['Sivaganga', 'Devakottai', 'Ilayangudi', 'Karaikudi', 'Manamadurai'],
        'Ramanathapuram': ['Ramanathapuram', 'Kadaladi', 'Kamuthi', 'Mudukulathur', 'Paramakudi'],
        'Virudhunagar': ['Virudhunagar', 'Aruppukkottai', 'Kariapatti', 'Rajapalayam', 'Sattur'],
        'Karur': ['Karur', 'Aravakurichi', 'Kadavur', 'Krishnarayapuram', 'Kulithalai'],
        'Namakkal': ['Namakkal', 'Kolli Hills', 'Kumarapalayam', 'Rasipuram', 'Tiruchengode'],
        'Dharmapuri': ['Dharmapuri', 'Harur', 'Karimangalam', 'Nallampalli', 'Palacode'],
        'Krishnagiri': ['Krishnagiri', 'Bargur', 'Denkanikottai', 'Hosur', 'Pochampalli'],
        'Tiruvannamalai': ['Tiruvannamalai', 'Arani', 'Cheyyar', 'Polur', 'Vandavasi']
    }
    
    crime_types = ['Theft', 'Cybercrime', 'Assault', 'Domestic Violence', 'Fraud', 'Burglary', 
                   'Drug Offense', 'Traffic Violation', 'Property Crime', 'Vandalism']
    
    area_types = ['Urban', 'Semi-Urban', 'Rural']
    weather_conditions = ['Sunny', 'Rainy', 'Cloudy', 'Hot']
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    months = list(range(1, 13))
    
    data = []
    
    for i in range(7000):
        district = random.choice(list(district_taluks.keys()))
        taluk = random.choice(district_taluks[district])
        
        # Urban districts have different crime patterns
        if district in ['Chennai', 'Coimbatore', 'Madurai']:
            area_type = random.choices(['Urban', 'Semi-Urban'], weights=[0.8, 0.2])[0]
            population_density = random.randint(800, 2000)
            unemployment_rate = random.uniform(3, 8)
            literacy_rate = random.uniform(85, 95)
            internet_penetration = random.uniform(70, 90)
            cctv_density = random.randint(15, 40)
            police_station_count = random.randint(8, 25)
        else:
            area_type = random.choices(['Rural', 'Semi-Urban', 'Urban'], weights=[0.6, 0.3, 0.1])[0]
            population_density = random.randint(200, 800)
            unemployment_rate = random.uniform(5, 15)
            literacy_rate = random.uniform(65, 85)
            internet_penetration = random.uniform(30, 70)
            cctv_density = random.randint(2, 15)
            police_station_count = random.randint(2, 8)
        
        # Crime type based on area characteristics
        if area_type == 'Urban' and internet_penetration > 60:
            crime_type = random.choices(crime_types, 
                                      weights=[2, 4, 1, 1, 3, 2, 1, 2, 1, 1])[0]
        elif unemployment_rate > 10:
            crime_type = random.choices(crime_types,
                                      weights=[4, 1, 2, 2, 2, 3, 1, 1, 2, 1])[0]
        else:
            crime_type = random.choice(crime_types)
        
        month = random.choice(months)
        day_of_week = random.choice(days)
        
        # Festival period affects crime patterns
        festival_period = 1 if month in [10, 11, 12, 4] else 0
        is_weekend = 1 if day_of_week in ['Saturday', 'Sunday'] else 0
        
        # Time patterns
        if crime_type in ['Theft', 'Burglary']:
            time_of_day = random.choices(['Night', 'Evening', 'Morning', 'Afternoon'], 
                                       weights=[4, 2, 1, 1])[0]
        elif crime_type == 'Cybercrime':
            time_of_day = random.choices(['Evening', 'Night', 'Afternoon', 'Morning'],
                                       weights=[3, 2, 2, 1])[0]
        else:
            time_of_day = random.choice(['Morning', 'Afternoon', 'Evening', 'Night'])
        
        weather = random.choice(weather_conditions)
        
        # Generate coordinates (approximate Tamil Nadu bounds)
        latitude = random.uniform(8.0, 13.5)
        longitude = random.uniform(76.0, 80.5)
        
        poverty_index = random.uniform(0.1, 0.8)
        past_crime_rate = random.uniform(0.5, 8.0)
        gender_ratio = random.uniform(0.9, 1.1)
        road_density = random.uniform(0.5, 5.0)
        education_index = literacy_rate / 100
        alcohol_availability = random.uniform(0.2, 0.9)
        transport_access = random.uniform(0.3, 0.95)
        public_event = random.choice([0, 1])
        age_group = random.choice(['18-25', '26-35', '36-45', '46-60', '60+'])
        
        data.append({
            'district': district,
            'taluk': taluk,
            'area_type': area_type,
            'latitude': round(latitude, 4),
            'longitude': round(longitude, 4),
            'crime_type': crime_type,
            'month': month,
            'day_of_week': day_of_week,
            'time_of_day': time_of_day,
            'population_density': population_density,
            'unemployment_rate': round(unemployment_rate, 2),
            'literacy_rate': round(literacy_rate, 2),
            'poverty_index': round(poverty_index, 3),
            'police_station_count': police_station_count,
            'cctv_density': cctv_density,
            'past_crime_rate': round(past_crime_rate, 2),
            'weather': weather,
            'festival_period': festival_period,
            'age_group': age_group,
            'gender_ratio': round(gender_ratio, 3),
            'road_density': round(road_density, 2),
            'education_index': round(education_index, 3),
            'internet_penetration': round(internet_penetration, 2),
            'alcohol_availability': round(alcohol_availability, 3),
            'transport_access': round(transport_access, 3),
            'public_event': public_event
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_crime_dataset()
    df.to_csv('data/raw/crime_tn_dataset.csv', index=False)
    print(f"Generated dataset with {len(df)} rows and {len(df.columns)} columns")
    print(df.head())