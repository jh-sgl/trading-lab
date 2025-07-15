from datetime import datetime, timedelta


def calc_days_to_expiration_KOSPI200(date: datetime.date) -> int:
    """
    Calculate days to the next KOSPI 200 futures expiration.
    Expiration is on the second Thursday of March, June, September, or December.
    """
    # Ensure input is datetime.date
    if isinstance(date, datetime):
        date = date.date()

    def second_thursday(year: int, month: int) -> datetime.date:
        """Return the second Thursday of a given month and year."""
        d = datetime(year, month, 1)
        thursdays = [
            d + timedelta(days=i)
            for i in range(14)
            if (d + timedelta(days=i)).weekday() == 3
        ]
        return thursdays[1].date()  # second Thursday

    # Find the next expiration month
    current_month = date.month
    current_year = date.year
    expiration_months = [3, 6, 9, 12]

    for m in expiration_months:
        if m > current_month or (
            m == current_month and date <= second_thursday(current_year, m)
        ):
            exp_date = second_thursday(current_year, m)
            break
    else:
        # All expiration months for this year have passed, go to next year's March
        exp_date = second_thursday(current_year + 1, 3)

    return (exp_date - date).days
