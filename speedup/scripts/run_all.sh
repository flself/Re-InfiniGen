for FIG in "16a" "16b" "17a"
do
  cd figure$FIG
  sh run.sh > log
  cd ..
done
