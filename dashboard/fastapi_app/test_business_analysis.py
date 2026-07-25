from business_analysis import (
    get_highest_revenue_segment,
    get_customer_count_by_segment,
    get_marketing_recommendation,
    get_segment_summary,
    load_customer_data,
)


try:
    customer_df = load_customer_data()

    print("\nDataset loaded successfully")
    print(customer_df.head())

    print("\nSegment summary")
    print(get_segment_summary(customer_df))

    print("\nHighest revenue segment")
    print(get_highest_revenue_segment(customer_df))

    print("\nCustomer count by segment")
    print(get_customer_count_by_segment(customer_df))

    print("\nAt Risk recommendation")
    print(get_marketing_recommendation("At Risk"))

except (FileNotFoundError, ValueError) as error:
    print(f"Error: {error}")