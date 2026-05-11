# Optimization of Hydrogen Transport Networks Under Environmental Constraints

## Overview

- **Optimization Type**: Multi-period Mixed Integer Linear Programming (MILP)
- **Objective**: Minimize total logistics cost (transportation + penalty for unmet demand)
- **Time Horizon**: 14 days
- **Ports**: 15 major ports across India and key international hubs
- **Solver**: PuLP with CBC solver

**Key Features**:
- Realistic maritime distances using `searoute` library
- Boil-off losses during storage
- Transit time delays for shipments
- Inventory balance and storage capacity constraints
- Production and demand balancing
- Penalty system for unmet demand

---

## Problem Statement

Efficient distribution of green hydrogen is critical for the global energy transition. This model optimizes the movement of hydrogen between production hubs and demand centers while managing inventory levels and minimizing losses due to boil-off and logistics inefficiencies.

---

## Model Components

### Physical & Operational Parameters
- **Ship Capacity**: 500 tonnes per shipment
- **Boil-off Rate**: 0.1% per day
- **Ship Speed**: 720 km/day
- **Initial Inventory**: 1000 tonnes at each port
- **Storage Capacity**: 2000 tonnes per port

### Ports Included
- **India**: Mumbai, Chennai, Kolkata, Kochi, Visakhapatnam, Paradip, Goa, Tuticorin, Haldia, Mormugao
- **International**: Singapore, Colombo, Jebel Ali, Port Klang, Shanghai

### Data Inputs
- Daily production at hubs (Mumbai, Jebel Ali, Singapore)
- Daily demand at consumption centers
- Real maritime route distances

---

## Technologies Used

- **Python**
- **PuLP** (Linear Programming)
- **NumPy**
- **searoute** (Maritime distance calculation)
- **Matplotlib** (Visualization)
- **Pandas** (Data handling)

---

## Results (14-Day Optimization)

**Optimization Status**: Optimal

**Key Performance Indicators**:

| Metric                            | Value                  |
|-----------------------------------|------------------------|
| **Total System Logistics Cost**   | ₹ 19,059,930.23       |
| **Total Hydrogen Transported**    | 25,070.42 tonnes      |
| **Average Cost per Tonne**        | ₹ 760.26              |
| **Total Hydrogen Produced**       | 20,300 tonnes         |
| **Total Boil-off Loss**           | 138.23 tonnes (0.68%) |
| **Total Demand**                  | 18,200 tonnes         |
| **Demand Satisfaction Rate**      | **92.30%**            |
| **Unmet Demand**                  | 1,400.80 tonnes (Shanghai) |

**Average Daily Boil-off**: 9.87 tonnes/day

---

## Visualizations

The model generates two key visualizations:
1. **Inventory Evolution** – Hydrogen inventory levels at each port over 14 days
2. **Shipment Heatmap** – Flow intensity across all shipping routes over time

---

## Project Structure
hydrogen-logistics-optimizer/
├── Multi-Period Hydrogen Maritime Network Optimization.ipynb                    # Main execution script
├── Multi-Period Hydrogen Maritime Network Optimization.py                   # python script
├── LICENSE/                      
├── results/                   # Output logs and plots
└── README.md
text---

## Author

**Aditya Singh**  
M.Sc. Applied Physics  
Amity University, Lucknow  

**Date**: February 2026

---

## Future Enhancements

- Integration of quantum optimization algorithms (QAOA) for routing subproblems
- Stochastic demand and production modeling
- Real-time weather and ocean current integration
- Multi-modal transportation (ship + rail/road)
- Larger time horizons and port networks

---

**This project demonstrates strong capabilities in mathematical optimization, supply chain modeling, and operations research applied to clean energy logistics.**
