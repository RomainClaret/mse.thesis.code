wget http://qa.mpi-inf.mpg.de/convex/data.zip
unzip data.zip -d $(pwd)/data/convex
mv $(pwd)/data/convex/data $(pwd)/data/convex/cache
mv $(pwd)/data/convex/cache/stopwords.txt $(pwd)/data/convex/
mv $(pwd)/data/convex/cache/identifier_predicates.json $(pwd)/data/convex/
rm data.zip

wget http://qa.mpi-inf.mpg.de/convex/ConvQuestions_test.zip
unzip ConvQuestions_test.zip -d $(pwd)/data/convex
rm ConvQuestions_test.zip

wget http://qa.mpi-inf.mpg.de/convex/ConvQuestions_dev.zip
unzip ConvQuestions_dev.zip -d $(pwd)/data/convex
rm ConvQuestions_dev.zip

wget http://qa.mpi-inf.mpg.de/convex/ConvQuestions_train.zip
unzip ConvQuestions_train.zip -d $(pwd)/data/convex
rm ConvQuestions_train.zip

mkdir $(pwd)/data/kb
wget http://gaia.infor.uva.es/hdt/wikidata/wikidata2018_09_11.hdt.gz
unzip wikidata2018_09_11.hdt.gz -d $(pwd)/data/kb
