from data import yfinanceuse
from riskanalysis import stablePPO




if __name__ == "__main__":
    yfinanceuse.main()
    stablePPO.main()
    data_path = (rf"weights\{yfinanceuse.get_tickersstr()}weights.csv")

