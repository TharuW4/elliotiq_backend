import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from . import main  # ඔයාගේ main.py එක මෙතනට import කරනවා

from .main import db
from .trade_audit import (
    get_trade_audit_stats,
    list_trade_audit_records,
    check_open_audit_trades
)

app = Flask(__name__)
CORS(app) # ඕනෑම තැනක ඉඳන් access කරන්න දෙනවා

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        
        # App එකෙන් එවන Data ටික ගන්න
        symbol = data.get('symbol', 'BTCUSDT')
        interval = data.get('interval', '15m')
        action = data.get('action', 'analyze') # analyze, status, scan, history
        market_type = data.get('market_type', 'crypto')
        
        # main.py එකේ අපි අලුතින් හදපු backend function එක call කරනවා
        result = main.run_analysis_backend(symbol, interval, action, market_type)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500


@app.route('/trade-audit/stats', methods=['GET'])
def trade_audit_stats():
    try:
        symbol = request.args.get('symbol')
        timeframe = request.args.get('timeframe')
        limit = int(request.args.get('limit', 1000))

        result = get_trade_audit_stats(
            db=db,
            symbol=symbol,
            timeframe=timeframe,
            limit=limit
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500


@app.route('/trade-audit/trades', methods=['GET'])
def trade_audit_trades():
    try:
        symbol = request.args.get('symbol')
        timeframe = request.args.get('timeframe')
        status = request.args.get('status')
        result_filter = request.args.get('result')
        limit = int(request.args.get('limit', 100))

        result = list_trade_audit_records(
            db=db,
            symbol=symbol,
            timeframe=timeframe,
            status=status,
            result=result_filter,
            limit=limit
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500


@app.route('/trade-audit/check-open', methods=['GET', 'POST'])
def trade_audit_check_open():
    try:
        market_type = request.args.get('market_type', 'crypto')
        limit = int(request.args.get('limit', 100))
        candle_period = request.args.get('period', '7d')

        result = check_open_audit_trades(
            db=db,
            market_type=market_type,
            limit=limit,
            candle_period=candle_period
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500


if __name__ == '__main__':
    # Cloud Run එකේදී PORT environment variable එකෙන් port එක ගන්නවා
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)