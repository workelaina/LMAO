i=0
while true
do
    ((i++))
    nvidia-smi > $i.log
    sleep 1
done
