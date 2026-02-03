import datetime 
import pytz
from typing import Optional,List,Tuple

def get_absolute_hour_of_year_with_dst(month: int, day: int, day_hour: int, year: int, timezone_name: str) -> int:
    """
    Calculates the absolute hour of the year (0-indexed) for a given date and hour,
    correctly handling Daylight Saving Time (DST) shifts.

    Args:
        month: The month (1-12).
        day: The day of the month (1-31).
        day_hour: The hour of the day (0-23).
        year: The year for the calculation.
        timezone_name: The IANA time zone string (e.g., 'Europe/Madrid', 'America/New_York').

    Returns:
        The absolute hour of the year (0-indexed).
    """
    try:
        # 1. Define the time zone
        tz = pytz.timezone(timezone_name)

        # 2. Create the start-of-year time (Jan 1st, 00:00), making it time zone-aware
        start_of_year = tz.localize(datetime(year, 1, 1, 0, 0, 0))

        # 3. Create the target time, making it time zone-aware
        # Note: localize handles ambiguous times (the hour repeated when clocks fall back)
        # by defaulting to the first occurrence (is_dst=False), which is generally safe.
        target_naive = datetime.datetime(year, month, day, day_hour)
        target_time = tz.localize(target_naive)

        # 4. Calculate the difference (timedelta)
        time_difference = target_time - start_of_year

        # 5. Convert the difference to total hours
        # This conversion correctly uses the total elapsed seconds, accounting for
        # the 23-hour or 25-hour days caused by DST shifts.
        total_hours = int(time_difference.total_seconds() / 3600)

        return total_hours

    except pytz.UnknownTimeZoneError:
        print(f"Error: Unknown time zone '{timezone_name}'.")
        return -1
    except ValueError as e:
        print(f"Error: Invalid date/time input. {e}")
        return -1




def datetime_to_hour_of_year(date_input: datetime.datetime) -> Optional[int]:
    """
    Calculates the absolute hour of the year (0-indexed) for a given date.
    It uses the timezone already attached to the input object for the calculation.

    Args:
        date_input: A timezone-aware datetime object (tzinfo MUST be set).

    Returns:
        The absolute hour of the year (0-indexed) as an integer, or None on error.
    """
    
    # 🚨 Input Validation: Check if the input is timezone-aware
    target_tz = date_input.tzinfo
    if target_tz is None or target_tz.utcoffset(date_input) is None:
        print("Error: Invalid date/time input. Input 'date_input' MUST be a timezone-aware datetime object.")
        return None

    try:
        # 1. The target timezone is extracted directly from the input object (target_tz)
        
        # 2. Create the start-of-year time (Jan 1st, 00:00) in the input's timezone
        start_of_year_naive = datetime.datetime(date_input.year, 1, 1, 0, 0, 0)
        
        # We must use localize() here since start_of_year_naive is, by definition, naive.
        start_of_year_aware = target_tz.localize(start_of_year_naive)

        # 3. The target time is already aware (date_input), so no conversion is needed.
        target_time_aware = date_input

        # 4. Calculate the difference (timedelta)
        # This calculation is safe because both datetimes are aware and use the same tzinfo.
        time_difference = target_time_aware - start_of_year_aware

        # 5. Convert the difference to total hours
        total_hours = int(time_difference.total_seconds() / 3600)

        return total_hours
    
    except pytz.exceptions.UnknownTimeZoneError:
        # This catch is mostly for safety, though the tzinfo should be valid if set by pytz initially.
        print(f"Error: Timezone attached to the input is not recognized.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    

def parse_datetime(date_string: str, timezone_name: str) -> Optional[datetime.datetime]:
    """
    Attempts to parse a date string using a predefined list of common formats.

    Args:
        date_string: The raw string input from the user (e.g., '2025-11-22 10:30').
        timezone_name: The timezone to apply (e.g., 'Europe/Madrid').

    Returns:
        A timezone-aware datetime object if successful, otherwise None.
    """
    
    # 1. Define the list of formats to try (from most specific/common to least)
    common_formats = [
        # Full date and time
        "%Y-%m-%d %H:%M:%S",  # 2025-11-22 10:30:00
        "%Y-%m-%d %H:%M",     # 2025-11-22 10:30
        "%d/%m/%Y %H:%M:%S",  # 22/11/2025 10:30:00
        "%d/%m/%Y %H:%M",     # 22/11/2025 10:30
        
        # Date only (assuming midnight)
        "%Y-%m-%d",           # 2025-11-22
        "%d/%m/%Y",           # 22/11/2025
        "%m/%d/%Y",           # 11/22/2025 (Common in US)
    ]

    # --- Timezone Setup ---
    try:
        # Get the timezone object using pytz
        tz = pytz.timezone(timezone_name)
    except pytz.exceptions.UnknownTimeZoneError:
        print(f"Error: Timezone '{timezone_name}' is not recognized.")
        return None
        
    # --- Iteration and Parsing ---
    for fmt in common_formats:
        try:
            # 1. Try to parse the string with the current format
            naive_dt = datetime.datetime.strptime(date_string, fmt)
            
            # 2. **Safety Net**: Ensure the object is naive before localization. 
            #    (Though strptime is expected to return a naive object, this is safe practice)
            naive_dt = naive_dt.replace(tzinfo=None)
            
            # 3. Localize the naive datetime object (make it timezone-aware)
            localized_dt = tz.localize(naive_dt)
            
            # 4. Success! Return the localized object immediately
            print(f"✅ Successfully parsed '{date_string}' using format: '{fmt}'")
            return localized_dt
            
        except ValueError:
            # If the parsing fails for this format, simply continue to the next one
            continue
            
    # If the loop finishes without returning, no format worked
    print(f"❌ Could not parse '{date_string}' with any known format.")
    return None


def list_full_hours_between(start_dt: datetime, end_dt: datetime) -> List[datetime]:
    """
    Lists all full hours (on the hour mark) between a start and end datetime.
    The list includes the hour of the start_dt, but excludes the hour of the end_dt.

    Args:
        start_dt: The starting datetime object (must be timezone-aware).
        end_dt: The ending datetime object (must be timezone-aware).

    Returns:
        A list of timezone-aware datetime objects, all set to the start of the hour.
    
    Raises:
        ValueError: If datetimes are not timezone-aware or start_dt is after end_dt.
    """
    
    # --- Input Validation ---
    if start_dt.tzinfo is None or start_dt.tzinfo.utcoffset(start_dt) is None:
        raise ValueError("Start datetime must be timezone-aware (tzinfo is required).")
    if end_dt.tzinfo is None or end_dt.tzinfo.utcoffset(end_dt) is None:
        raise ValueError("End datetime must be timezone-aware (tzinfo is required).")
    if start_dt >= end_dt:
        return []

    # --- 1. Find the first full hour mark to start the iteration ---
    # Set the minute and second to zero for the start_dt's hour
    current_dt = start_dt.replace(minute=0, second=0, microsecond=0)
    
    # If the start_dt itself was not exactly on the hour, we must include it
    # as the first item in the list, otherwise, the next iteration would skip it.
    if start_dt.minute > 0 or start_dt.second > 0:
        pass # current_dt now represents the start of the hour containing start_dt
        
    hours_list = []
    
    # --- 2. Iterate hour by hour until the end_dt is reached ---
    one_hour = datetime.timedelta(hours=1)
    
    while current_dt < end_dt:
        # Append the current full hour to the list
        hours_list.append(current_dt)
        
        # Move to the next hour
        current_dt += one_hour
        
    return hours_list

def absolute_hours_list(date_ini,date_end,time_zone='Europe/Paris'):

    result_ini = parse_datetime(date_ini, time_zone)
    result_end = parse_datetime(date_end, time_zone)

    hours_list = []
    for item in list_full_hours_between(result_ini,result_end):
        hours_list += [datetime_to_hour_of_year(item)]

    return hours_list

        

def hour_index_to_date_components(absolute_hour_index: int, timezone_name: str) -> Optional[Tuple[int, int]]:
    """
    Converts an absolute hour of the year index (0-indexed) to the corresponding 
    day of the year and hour of the day, respecting DST rules.

    Args:
        absolute_hour_index: The 0-indexed hour of the year (e.g., 8759 is the last hour).
        timezone_name: The IANA time zone string (e.g., 'Europe/Madrid').

    Returns:
        A tuple (day_of_year, hour_of_day) as integers, or None on error.
    """
    
    current_year = datetime.datetime.now().year # Use the current year for the calculation
    
    try:
        # 1. Define the time zone
        tz = pytz.timezone(timezone_name)

        # 2. Create the start-of-year time (Jan 1st, 00:00)
        # This is the reference point for hour index 0.
        start_of_year_naive = datetime.datetime(current_year, 1, 1, 0, 0, 0)
        
        # Localize the start of the year to make it aware
        start_of_year_aware = tz.localize(start_of_year_naive)

        # 3. Create the time difference
        # Multiplying the hour index by 3600 gives the exact number of seconds 
        # that have elapsed since the start_of_year_aware.
        time_difference = datetime.timedelta(hours=absolute_hour_index)

        # 4. Calculate the target time
        # This addition operation correctly handles DST shifts. If 25 hours have
        # passed since Jan 1st, and one hour was lost due to DST, the target time 
        # will only be advanced by 24 clock hours.
        target_time_aware = start_of_year_aware + time_difference
        
        # 5. Extract the required components
        # %j gives the 1-indexed day of the year (1-366).
        day_of_year = int(target_time_aware.strftime('%j'))
        hour_of_day = target_time_aware.hour # 0-23
        
        return (day_of_year, hour_of_day)
    
    except pytz.exceptions.UnknownTimeZoneError:
        print(f"Error: Timezone '{timezone_name}' is not recognized.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    

def day_to_absolute_hour(day_of_year: int, target_year: int, timezone_name: str = 'Europe/Madrid') -> int:
    """
    Calculates the absolute hour of the year (based on UTC) for a specific day 
    at 00:00 local time, correctly accounting for time zone and DST changes.

    Args:
        day_of_year: The sequential day of the year (1 for Jan 1st, 365/366 for Dec 31st).
        target_year: The year for the calculation (e.g., 2024).
        timezone_name: The target time zone string (e.g., 'Europe/London', 'America/New_York'). 
                       Defaults to 'UTC'.

    Returns:
        The total number of hours elapsed from the beginning of the year 
        (Jan 1st 00:00 UTC) up to the specified date and time (converted to UTC).
    
    Raises:
        pytz.UnknownTimeZoneError: If the provided timezone_name is invalid.
    """
    
    # 1. Define the start of the year and the target date (unlocalized)
    start_of_year = datetime.datetime(target_year, 1, 1, 0, 0, 0)
    
    # Calculate the target date (00:00 local time)
    # We subtract 1 from day_of_year because Jan 1st is day 1, but represents 0 days of offset.
    target_date = start_of_year + datetime.timedelta(days=day_of_year - 1)

    try:
        # 2. Localize the dates
        tz = pytz.timezone(timezone_name)
        
        # Localize the date to the target timezone (this handles DST shift at 00:00 if any)
        # .localize() is the safe way to assign timezone awareness.
        start_of_year_local = tz.localize(start_of_year)
        target_date_local = tz.localize(target_date)

        # 3. Convert both localized times to UTC
        start_of_year_utc = start_of_year_local.astimezone(pytz.utc)
        target_date_utc = target_date_local.astimezone(pytz.utc)

        # 4. Calculate the difference in hours
        time_difference = target_date_utc - start_of_year_utc
        
        # Convert total seconds in the timedelta to total hours (integer result)
        absolute_hour = int(time_difference.total_seconds() / 3600)
        
        return absolute_hour

    except pytz.UnknownTimeZoneError:
        raise pytz.UnknownTimeZoneError(f"Time zone '{timezone_name}' is not recognized by pytz.")
    
if __name__ == '__main__':

    result_1 = parse_datetime('31/12/2024 23:00', 'Europe/Paris')
    print(datetime_to_hour_of_year(result_1))

    result_ini = parse_datetime( '1/12/2024 23:00', 'Europe/Paris')
    result_end = parse_datetime('31/12/2024 23:00', 'Europe/Paris')

    print(list_full_hours_between(result_ini,result_end))

    print(absolute_hours_list('1/1/2024 00:00','31/1/2024 23:00'))

