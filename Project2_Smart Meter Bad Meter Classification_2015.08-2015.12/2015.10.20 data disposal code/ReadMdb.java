package meter;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.Statement;
import java.util.Properties;
public class ReadMdb {
	
		/**
		 * TODO : 读取文件access
		 * 
		 * @param filePath
		 * @return
		 * @throws ClassNotFoundException
		 */
		public static void readFileACCESS(File mdbFile) {
			Properties prop = new Properties();
			prop.put("charSet", "gb2312"); // 这里是解决中文乱码
			prop.put("user", "");
			prop.put("password", "");
			String url = "jdbc:odbc:driver={Microsoft Access Driver (*.mdb)};DBQ="
					+ mdbFile.getAbsolutePath();
			Statement stmt = null;
			ResultSet rs = null;
			String tableName = null;
			try {
				Class.forName("sun.jdbc.odbc.JdbcOdbcDriver");
				// 连接到mdb文件
				Connection conn = DriverManager.getConnection(url, prop);
				ResultSet tables = conn.getMetaData().getTables(
						mdbFile.getAbsolutePath(), null, null,
						new String[] { "TABLE" });
				// 获取第一个表名
				if (tables.next()) {
					tableName = tables.getString(3);// getXXX can only be used once
				} else {
					return;
				}
				stmt = (Statement) conn.createStatement();
				// 读取第一个表的内容
				rs = stmt.executeQuery("select * from " + tableName);
				ResultSetMetaData data = rs.getMetaData();
				while (rs.next()) {
					for (int i = 1; i <= data.getColumnCount(); i++) {
						System.out.print(rs.getString(i) + "	");
					}
					System.out.println();
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		public static void main(String[] args) throws IOException {
			//readFileACCESS(new File("D:/data/meter/smart meter materials/smart meter materials/Mtr_Temp_study10202014.mdb"));
			BufferedWriter bw = null;

			try {
				bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("D:/data/meter/smart meter materials/smart meter materials/loaddata.csv"),"utf-8"));
			} catch (UnsupportedEncodingException | FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			    try
			    {

			        Connection conn=DriverManager.getConnection("jdbc:ucanaccess://D:/data/meter/smart meter materials/smart meter materials/Mtr_Temp_study10202014.mdb");
			        Statement stment = conn.createStatement();
			       
			        String qry = "SELECT * FROM LoadData";

			        ResultSet rs = stment.executeQuery(qry);
			        bw.append("id,meter_id,location,date,loaddata");
			        bw.newLine();
			        while(rs.next())
			        {
			            String id    = rs.getString("ID") ;
			            String meter = rs.getString("Meter");
			            String town = rs.getString("Town");
			            String it = rs.getString("IntTime").substring(0, 19);
			            String iv = rs.getString("IntVal");

			           // System.out.println(id +","+ meter+","+town+","+it+","+iv);
			            bw.append(id +",\""+ meter+"\",\""+town+"\","+it+","+iv);
			            bw.newLine();
			        }
			    }
			    catch(Exception err)
			    {
			        System.out.println(err);
			    }
			    bw.close();
			    //System.out.println("Hasith Sithila");

			}
	}
