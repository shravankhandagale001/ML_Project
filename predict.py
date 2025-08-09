# predict.py (updated)
import argparse
from finale import SolarEnergyPredictor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hour', type=int, required=True)
    parser.add_argument('--month', type=int, required=True)
    parser.add_argument('--temp', type=float, required=True)
    parser.add_argument('--zenith', type=float, required=True)
    parser.add_argument('--ghi', type=float, required=True)
    parser.add_argument('--humidity', type=float, required=True)
    parser.add_argument('--wind', type=float, required=True)
    parser.add_argument('--panel_efficiency', type=float, default=0.20,
                   help='Solar panel efficiency (0-1, default: 0.20)')
    parser.add_argument('--system_efficiency', type=float, default=0.85,
                    help='System efficiency including inverter losses (0-1, default: 0.85)')
    parser.add_argument('--panel_count', type=int, default=1,
                    help='Number of solar panels (default: 1)')
    parser.add_argument('--panel_size', type=float, default=1.7,
                    help='Panel size in m² (default: 1.7)')
    args = parser.parse_args()

    model = SolarEnergyPredictor()
    model.load_model('solar_energy_model.pkl')

    result = model.predict_from_user_input({
        'Hour': args.hour,
        'Month': args.month,
        'Temperature': args.temp,
        'Solar Zenith Angle': args.zenith,
        'GHI': args.ghi,
        'Relative Humidity': args.humidity,
        'Wind Speed': args.wind
    })
    print(f"\nPredicted Solar Energy: {result['predictions'][0]:.2f} kWh")

if __name__ == "__main__":
    main()