REM Commented configs were tried bud didn't work well
REM python uppercase.py 
REM python uppercase.py --learning_rate 0.005
REM python uppercase.py --dropout 0.4 --layers 5
REM python uppercase.py --dropout 0.6 --layers 5
REM python uppercase.py --window 10

REM python uppercase.py --learning_rate 0.002 --learning_rate_final 0.00002 --dropout 0.6 --layers 5
REM python uppercase.py --learning_rate 0.002 --learning_rate_final 0.00002 --dropout 0.6 --layers 5 --batch_size 512
REM python uppercase.py --learning_rate 0.001 --learning_rate_final 0.00002 --dropout 0.6 --layers 5 --batch_size 512
REM python uppercase.py --learning_rate 0.001 --learning_rate_final 0.0001 --dropout 0.6 --layers 5 --batch_size 512

REM python uppercase.py --learning_rate 0.001 --learning_rate_final 0.00002 --dropout 0.6 --layers 5 --batch_size 4096
REM python uppercase.py --learning_rate 0.001 --learning_rate_final 0.0001 --dropout 0.6 --layers 5 --batch_size 4096

REM python uppercase.py --learning_rate 0.001 --learning_rate_final 0.0001 --dropout 0.6 --layers 5 --batch_size 4096 --window 7
REM python uppercase.py --learning_rate 0.001 --learning_rate_final 0.0001 --dropout 0.8 --layers 5 --batch_size 4096 --window 7
REM python uppercase.py --learning_rate 0.001 --learning_rate_final 0.0001 --dropout 0.8 --layers 7 --batch_size 4096 --window 7

REM python uppercase.py --learning_rate 0.001 --learning_rate_final 0.0001 --dropout 0.6 --layers 4 --batch_size 512
REM python uppercase.py --learning_rate 0.001 --learning_rate_final 0.0001 --dropout 0.6 --layers 5 --batch_size 512 --window 4

REM python uppercase.py --learning_rate 0.001 --learning_rate_final 0.0001 --dropout 0.6 --layers 5 --batch_size 512 --activation sigmoid
REM python uppercase.py --learning_rate 0.001 --learning_rate_final 0.0001 --dropout 0.6 --layers 5 --batch_size 512 --activation tanh
REM python uppercase.py --learning_rate 0.001 --learning_rate_final 0.0001 --dropout 0.6 --layers 7 --batch_size 512 --activation tanh

python uppercase.py --learning_rate 0.001 --learning_rate_final 0.0001 --dropout 0.6 --layers 5 --batch_size 512 --activation relu --epochs 3