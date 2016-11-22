import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class PageRank {
	
	static int maxIter = 10;		//stop criteria is max iterations count = 10
	static DoubleWritable[] prArray = new DoubleWritable[101];		//array to store page rank, element at index 0 won't be used
	static int iterCnt;
	static final double DAMPING_FACTOR = 0.85;

	public static void main(String[] args) throws Exception {
			
		for (int s = 0; s < 101; s++) {			//initialize page rank array with initial values of 1.0;
			prArray[s] = new DoubleWritable(1.0);
		}
		
		iterCnt = 0;
		while (iterCnt < maxIter) {
			iterCnt++;	
			Configuration conf = new Configuration();
			Job job = Job.getInstance(conf, "page rank");
			job.setJarByClass(PageRank.class);
			job.setMapperClass(PRMapper.class);
			job.setReducerClass(PRReducer.class);
			job.setOutputKeyClass(IntWritable.class);
			job.setOutputValueClass(DoubleWritable.class);
			
			//check page rank for debugging purposes
			System.out.println("Iteration: " + iterCnt);
			
			//these are the page ranks before each iteration
			for (int k = 0; k < 101; k++) {			
				System.out.println(prArray[k].toString());
			}
							
			String outDir = args[1] + "iter" + iterCnt;		//setup output folder to store result after each iteration
			FileInputFormat.addInputPath(job, new Path(args[0]));
			FileSystem fs = FileSystem.get(conf);			
			//check whether output folder exists. If so, delete it.	
			if (fs.exists(new Path(outDir))) {			
				fs.delete(new Path(outDir),true);
			}		
			FileOutputFormat.setOutputPath(job, new Path(outDir));
			job.waitForCompletion(true);
		}
		System.exit(0);			
	}
		
	public static class PRMapper
				extends Mapper<LongWritable, Text, IntWritable, DoubleWritable> {
		
		IntWritable mOutKey = new IntWritable();
		DoubleWritable mOutValue = new DoubleWritable();
		
		public void map(LongWritable key, Text value, Context context) 
				throws IOException, InterruptedException {
			
			String strValue = value.toString().replaceAll(",", "\t");
			String[] split = strValue.split("\t");
			double[] tmpArray = new double[101];		//array to temporarily store page rank
			int nodIdx = Integer.parseInt(split[0]);
			
			for (int j = 0; j < 101; j++) {			//initialize temporary array with initial values of 0;
				tmpArray[j] = 0;
			}
			
			for (int i = 1; i < split.length; i++) {
				int onodIdx = Integer.parseInt(split[i]);
				tmpArray[onodIdx] = prArray[nodIdx].get()/(split.length - 1); 
				mOutKey.set(onodIdx);
				mOutValue.set(tmpArray[onodIdx]);
				context.write(mOutKey, mOutValue);
			}
		}
	}


	public static class PRReducer
				extends Reducer<IntWritable, DoubleWritable, IntWritable, DoubleWritable> {

		IntWritable outKey = new IntWritable();
		DoubleWritable outValue = new DoubleWritable();
		
		public void reduce(IntWritable key, Iterable<DoubleWritable> values, Context context) 
				throws IOException, InterruptedException {
			
			int nId = key.get();
			double sum = 0;
			
			for (DoubleWritable val : values) {
				sum += val.get();
			}
			
			prArray[nId].set(1 - DAMPING_FACTOR + DAMPING_FACTOR * sum);
			outKey.set(nId);
			outValue.set(prArray[nId].get());
			context.write(outKey, outValue);
		}
	}
}
